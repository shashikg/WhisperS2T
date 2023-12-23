import whisper
from whisper.decoding import DecodingOptions

from .. import WhisperModel
from ...configs import *


ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": True,
    "return_no_speech_prob": True,
    "patience": 1,
    "length_penalty": 1,
}


class WhisperModelOAI(WhisperModel):
    def __init__(self,
                 model_name: str,
                 device="cuda",
                 compute_type="float16",
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 asr_options={},
                 **model_kwargs):

        self.model_name = model_name
        self.asr_options = ASR_OPTIONS
        self.asr_options.update(asr_options)

        self.model = whisper.load_model(model_name)
        self.model.to(device).eval()
        
        self.decode_options = {
            "sample_len": max_text_token_len,
            'fp16': True if compute_type == "float16" else False
        }

        for k, v in self.asr_options.items():
            if hasattr(DecodingOptions, k):
                self.decode_options[k] = v

        super().__init__(
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_decode_options(self, params={}):
        self.decode_options.update(params)

        if 'sample_len' in params:
            self.update_params(params={'max_text_token_len': params['sample_len']})
    
    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata):

        if self.compute_type == "float16":
            features = features.to(self.device).half()

        lang_and_task_pairs = {}
        for _i, _p in enumerate(prompts):
            try:
                lang_and_task_pairs[(_p[-3], _p[-2])].append(_i)
            except:
                lang_and_task_pairs[(_p[-3], _p[-2])] = [_i]


        response = [{} for _ in prompts]
        for (task, lang), idx_list in lang_and_task_pairs.items():
            
            results = self.model.decode(features[idx_list].to(self.device), DecodingOptions(task=task, language=lang, **self.decode_options))

            for idx, result in zip(idx_list, results):
                response[idx]['text'] = result.text.strip()

                if self.asr_options['return_scores']:
                    response[idx]['avg_logprob'] = result.avg_logprob

                if self.asr_options['return_no_speech_prob']:
                    response[idx]['no_speech_prob'] = result.no_speech_prob

        return response