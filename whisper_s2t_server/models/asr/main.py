import os
import time
import torch
import argparse

from whisper_s2t.configs import *
from whisper_s2t.audio import LogMelSpectogram, pad_or_trim

from . import pullTask, finishTask
from ...logger import Logger

ASR_WAIT_TIME = float(os.getenv("ASR_WAIT_TIME", "0.1"))
ASR_BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "32"))


def get_server_backend(backend='ct2'):
    if backend.lower() in ["tensorrt", "trt", "trt-llm", "tensorrt-llm", "trt_llm", "tensorrt_llm"]:
        from whisper_s2t.backends.tensorrt.model import WhisperModelTRT as WhisperModel
    else:
        from whisper_s2t.backends.ctranslate2.model import WhisperModelCT2 as WhisperModel

    class WhisperModelServer(WhisperModel):
        def _init_dependables(self):
            # Rescaled Params
            self.dta_padding = int(self.dta_padding*SAMPLE_RATE)
            self.max_initial_prompt_len = self.max_text_token_len//2 -1

            # Load Pre-Processor
            self.preprocessor = LogMelSpectogram(n_mels=self.n_mels).to(self.device)

        def data_collate_fn(self, batch):

            if self.use_dynamic_time_axis:
                max_len = min(max([_['seq_len'] for _ in batch]) + self.dta_padding, N_SAMPLES)
            else:
                max_len = N_SAMPLES

            signal_batch = torch.stack([torch.from_numpy(pad_or_trim(_['audio'], length=max_len)).to(self.device) for _ in batch])
            seq_len = torch.tensor([_['seq_len'] for _ in batch]).to(self.device)

            for _ in batch:
                if _['initial_prompt']:
                    initial_prompt = " " + _['initial_prompt'].strip()
                    _['initial_prompt_tokens'] = self.tokenizer.encode(_['initial_prompt'])[-self.max_initial_prompt_len:]
                else:
                    _['initial_prompt_tokens'] = []

                _['task_prompt'] = self.tokenizer.sot_sequence(task=_['task'], lang=_['lang'])
                
                if self.without_timestamps:
                    _['task_prompt'].append(self.tokenizer.no_timestamps)
                else:
                    _['task_prompt'].append(self.tokenizer.timestamp_begin)


            prompt_batch = []
            initial_prompt_max_len = max([len(_['initial_prompt_tokens']) for _ in batch])
            if initial_prompt_max_len:
                for _ in batch: 
                    _prompt = [self.tokenizer.sot_prev] + (initial_prompt_max_len-len(_['initial_prompt_tokens']))*[self.tokenizer.silent_token]
                    _prompt = _prompt + _['initial_prompt_tokens'] + _['task_prompt']
                    prompt_batch.append(_prompt)
            else:
                for _ in batch: 
                    prompt_batch.append(_['task_prompt'])

            return signal_batch, prompt_batch, seq_len
        
    return WhisperModelServer
            

def data_loader():
    batched_req = []
    batch_start_time = time.time()
    while True:
        if len(batched_req) == ASR_BATCH_SIZE:
            yield batched_req
            batched_req = []
        elif len(batched_req) and ((time.time()-batch_start_time) >= ASR_WAIT_TIME):
            yield batched_req
            batched_req = []
        else:
            job_id, job_data = pullTask()

            if job_id is None:
                time.sleep(0.2*ASR_WAIT_TIME)
            else:
                batched_req.append({'job_id': job_id, 'seq_len': job_data['audio'].shape[-1], **job_data})

                if len(batched_req) == 1:
                    batch_start_time = time.time()


def start_worker(args):
    Logger.info(f"Starting ASR Model!")

    WhisperModelServer = get_server_backend(backend=args.backend])
    compute_type = 'float16' if args.device == 'cuda' else 'float32'
    ASR_MODEL = WhisperModelServer(args.model_identifier, 
                                   device=args.device, 
                                   compute_type=compute_type)

    Logger.info(f"Started ASR Model!")

    for batch in data_loader():
        Logger.info(f"[ASR] Processing #{len(batch)} requests!")

        signal_batch, prompt_batch, seq_len = ASR_MODEL.data_collate_fn(batch)

        mels, seq_len = ASR_MODEL.preprocessor(signal_batch, seq_len)
        result = ASR_MODEL.generate_segment_batched(mels.to(ASR_MODEL.device), prompt_batch, seq_len, None)

        for _req, _res in zip(batch, result):
            finishTask(_req['job_id'], _res)

        Logger.info(f"[ASR] Processed #{len(batch)} requests!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='whisper_s2t_server_asr_worker')
    parser.add_argument('--model_identifier', default="tiny", help='Model name to use')
    parser.add_argument('--backend', default='ct2', help='Which backend to use')
    parser.add_argument('--device', default='cpu', help='cpu/cuda')

    args = parser.parse_args()
    start_worker(args)