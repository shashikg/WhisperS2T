import os
import tokenizers
import ctranslate2
import numpy as np

from .tokenizer import Tokenizer
from .hf_utils import download_model


from .. import WhisperModel
from ...configs import *


FAST_ASR_OPTIONS = {
    "beam_size": 1,
    "best_of": 1, # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
}


BEST_ASR_CONFIG = {
    "beam_size": 5,
    "best_of": 1, # Placeholder
    "patience": 2,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
}


class WhisperModelCT2(WhisperModel):
    def __init__(self,
                 model_name_or_path: str,
                 cpu_threads=4,
                 num_workers=1,
                 device="cuda",
                 device_index=0,
                 compute_type="float16",
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 asr_options={},
                 **model_kwargs):

        
        # Get local model path or download from huggingface
        if os.path.isdir(model_name_or_path):
            self.model_path = model_name_or_path
        else:
            self.model_path = download_model(model_name_or_path)
        
        # Load model
        self.model = ctranslate2.models.Whisper(self.model_path,
                                                device=device,
                                                device_index=device_index,
                                                compute_type=compute_type,
                                                intra_threads=cpu_threads,
                                                inter_threads=num_workers)
        
        # Load tokenizer
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        tokenizer = Tokenizer(tokenizers.Tokenizer.from_file(tokenizer_file), self.model.is_multilingual)

        # ASR Options
        self.asr_options = FAST_ASR_OPTIONS
        self.asr_options.update(asr_options)
        
        self.generate_kwargs = {
            "max_length": max_text_token_len,
            "return_scores": self.asr_options['return_scores'],
            "return_no_speech_prob": self.asr_options['return_no_speech_prob'],
            "length_penalty": self.asr_options['length_penalty'],
            "repetition_penalty": self.asr_options['repetition_penalty'],
            "no_repeat_ngram_size": self.asr_options['no_repeat_ngram_size'],
            "beam_size": self.asr_options['beam_size'],
            "patience": self.asr_options['patience'],
            "suppress_blank": self.asr_options['suppress_blank'],
            "suppress_tokens": self.asr_options['suppress_tokens'],
            "max_initial_timestamp_index": int(round(self.asr_options['max_initial_timestamp']/TIME_PRECISION)),
            "sampling_temperature": self.asr_options['sampling_temperature'],
        }

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if 'max_text_token_len' in params:
            self.update_params(params={'max_text_token_len': params['max_text_token_len']})
    
    def encode(self, features):
        """
        [Not Used]
        """
        
        features = ctranslate2.StorageView.from_array(features.contiguous())
        return self.model.encode(features)
    
    def generate_segment_batched(self, features, prompts):
        
        if self.device == 'cpu':
            features = np.ascontiguousarray(features.detach().numpy())
        else:
            features = features.contiguous()

        result = self.model.generate(ctranslate2.StorageView.from_array(features),
                                     prompts,
                                     **self.generate_kwargs)
        
        text = self.tokenizer.decode_batch([x.sequences_ids[0] for x in result])
        
        response = []
        for idx, r in enumerate(result):
            response.append({'text': text[idx].strip()})

            if self.generate_kwargs['return_scores']:
                seq_len = len(r.sequences_ids[0])
                cum_logprob = r.scores[0]*(seq_len**self.generate_kwargs['length_penalty'])
                response[-1]['avg_logprob'] = cum_logprob/(seq_len + 1)

            if self.generate_kwargs['return_no_speech_prob']:
                response[-1]['no_speech_prob'] = r.no_speech_prob

        return response