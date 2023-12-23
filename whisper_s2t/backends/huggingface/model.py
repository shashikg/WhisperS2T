import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .. import WhisperModel
from ...configs import *


ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": False,
    "return_no_speech_prob": False,
    "use_flash_attention": True,
    "use_better_transformer": False,
}


COMPUTE_TYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16
}


class WhisperModelHF(WhisperModel):
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

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, 
                                                                     torch_dtype=COMPUTE_TYPE_TO_TORCH_DTYPE.get(compute_type, torch.float32), 
                                                                     low_cpu_mem_usage=True, 
                                                                     use_safetensors=True,
                                                                     use_flash_attention_2=self.asr_options["use_flash_attention"])
        self.model.config.forced_decoder_ids = None
        self.model.to(device).eval()

        if self.asr_options["use_better_transformer"]:
            self.model = self.model.to_bettertransformer()
        
        self.generate_kwargs = {
            "max_new_tokens": max_text_token_len,
            "num_beams": self.asr_options['beam_size'],
            "return_timestamps": not self.asr_options['without_timestamps'],
        }

        super().__init__(
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if 'max_new_tokens' in params:
            self.update_params(params={'max_text_token_len': params['max_new_tokens']})
    
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
            predicted_ids = self.model.generate(features[idx_list], 
                                                task=task,
                                                language=lang,
                                                **self.generate_kwargs)
        
            results = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for idx, text in zip(idx_list, results):
                response[idx]['text'] = text.strip()

        return response