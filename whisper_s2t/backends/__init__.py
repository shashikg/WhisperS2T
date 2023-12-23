import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from ..configs import *
from ..data import WhisperDataLoader
from ..audio import LogMelSpectogram
from ..speech_segmenter import SpeechSegmenter


class NoneTokenizer:
    def __init__(self):
        self.sot_prev = 0
        self.silent_token = 0
        self.no_timestamps = 0
        self.timestamp_begin = 0
    
    def sot_sequence(self, task=None, lang=None):
        return [task, lang]

    def encode(self, text):
        return [0]


class WhisperModel(ABC):
    def __init__(self,
                 tokenizer=None,
                 vad_model=None,
                 n_mels=80,
                 device="cuda",
                 device_index=0,
                 compute_type="float16",
                 merge_chunks=True,
                 dta_padding=3.0,
                 use_dynamic_time_axis = False,
                 max_speech_len=29.0,
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 without_timestamps=True,
                 speech_segmenter_options={}):
        
        # Configure Params
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type

        self.n_mels = n_mels
        self.merge_chunks = merge_chunks
        self.max_speech_len = max_speech_len

        self.dta_padding = dta_padding
        self.use_dynamic_time_axis = use_dynamic_time_axis

        self.without_timestamps = without_timestamps
        self.max_text_token_len = max_text_token_len

        self.vad_model = vad_model
        self.speech_segmenter_options = speech_segmenter_options
        self.speech_segmenter_options['max_seg_len'] = self.max_speech_len

        # Tokenizer
        if tokenizer is None:
            tokenizer = NoneTokenizer()

        self.tokenizer = tokenizer

        self._init_dependables()


    def _init_dependables(self):
        # Rescaled Params
        self.dta_padding = int(self.dta_padding*SAMPLE_RATE)
        self.max_initial_prompt_len = self.max_text_token_len//2 -1

        # Load Pre Processor
        self.preprocessor = LogMelSpectogram(n_mels=self.n_mels).to(self.device)

        # Load Speech Segmenter
        self.speech_segmenter = SpeechSegmenter(self.vad_model, device=self.device, **self.speech_segmenter_options)

        # Load Data Loader
        self.data_loader = WhisperDataLoader(
            self.device, self.tokenizer, self.speech_segmenter, 
            dta_padding=self.dta_padding,
            without_timestamps=self.without_timestamps, 
            max_speech_len=self.max_speech_len, 
            max_initial_prompt_len=self.max_initial_prompt_len, 
            use_dynamic_time_axis=self.use_dynamic_time_axis,
            merge_chunks=self.merge_chunks
        )

    def update_params(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)
        
        self._init_dependables()

    
    @abstractmethod
    def generate_segment_batched(self, features, prompts):
        pass
        
    @torch.no_grad()
    def transcribe(self, audio_files, lang_codes=None, tasks=None, initial_prompts=None, batch_size=8):
        
        if lang_codes == None:
            lang_codes = len(audio_files)*['en']
            
        if tasks == None:
            tasks = len(audio_files)*['transcribe']
        
        if initial_prompts == None:
            initial_prompts = len(audio_files)*[None]
            
        responses = []
        for signals, prompts, seq_len in self.data_loader(audio_files, lang_codes, tasks, initial_prompts, batch_size=batch_size, use_vad=False):
            mels, seq_len = self.preprocessor(signals, seq_len)
            res = self.generate_segment_batched(mels.to(self.device), prompts)
            responses.extend(res)
        
        return responses
    
    @torch.no_grad()
    def transcribe_with_vad(self, audio_files, lang_codes=None, tasks=None, initial_prompts=None, batch_size=8):
        
        if lang_codes == None:
            lang_codes = len(audio_files)*['en']
            
        if tasks == None:
            tasks = len(audio_files)*['transcribe']
        
        if initial_prompts == None:
            initial_prompts = len(audio_files)*[None]
            
        responses = [[] for _ in audio_files]
        
        pbar_pos = 0
        with tqdm(total=len(audio_files)*100, desc=f"Transcribing") as pbar:
            for signals, prompts, seq_len, seg_metadata, pbar_update in self.data_loader(audio_files, lang_codes, tasks, initial_prompts, batch_size=batch_size):
                mels, seq_len = self.preprocessor(signals, seq_len)
                res = self.generate_segment_batched(mels.to(self.device), prompts, seq_len, seg_metadata)

                for res_idx, _seg_metadata in enumerate(seg_metadata):
                    responses[_seg_metadata['file_id']].append({**res[res_idx],
                                                                'start_time': round(_seg_metadata['start_time'], 3),
                                                                'end_time': round(_seg_metadata['end_time'], 3)})
                
                if (pbar_pos) <= pbar.total:
                    pbar_pos += pbar_update
                    pbar.update(pbar_update)
            
            pbar.update(pbar.total-pbar_pos)
        
        return responses