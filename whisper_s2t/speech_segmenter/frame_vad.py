import os
import torch
import numpy as np

from . import VADBaseClass
from .. import BASE_PATH


class FrameVAD(VADBaseClass):
    def __init__(self, 
                 device=None,
                 chunk_size=15.0,
                 margin_size=1.0,
                 frame_size=0.02,
                 batch_size=4,
                 sampling_rate=16000):
        
        super().__init__(sampling_rate=sampling_rate)
        
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.device = device
        
        if self.device == 'cpu':
            # This is a JIT Scripted model of Nvidia's NeMo Framewise Marblenet Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_frame_marblenet
            self.vad_pp = torch.jit.load(os.path.join(BASE_PATH, "assets/vad_pp_cpu.ts")).to(self.device)
            self.vad_model = torch.jit.load(os.path.join(BASE_PATH, "assets/frame_vad_model_cpu.ts")).to(self.device)
        else:
            self.vad_pp = torch.jit.load(os.path.join(BASE_PATH, "assets/vad_pp_gpu.ts")).to(self.device)
            self.vad_model = torch.jit.load(os.path.join(BASE_PATH, "assets/frame_vad_model_gpu.ts")).to(self.device)

        self.vad_pp.eval()
        self.vad_model.eval()
        
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.chunk_size = chunk_size
        self.margin_size = margin_size
        
        self._init_params()
        
    def _init_params(self):
        self.signal_chunk_len = int(self.chunk_size*self.sampling_rate)
        self.signal_stride = int(self.signal_chunk_len-2*int(self.margin_size*self.sampling_rate))
        
        self.margin_logit_len = int(self.margin_size/self.frame_size)
        self.signal_to_logit_len = int(self.frame_size*self.sampling_rate)
        
        self.vad_pp.to(self.device)
        self.vad_model.to(self.device)
        
    def update_params(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)
        
        self._init_params()
    
    def prepare_input_batch(self, audio_signal):
        input_signal = []
        input_signal_length = []
        for s_idx in range(0, len(audio_signal), self.signal_stride):
            _signal = audio_signal[s_idx:s_idx+self.signal_chunk_len]
            _signal_len = len(_signal)
            input_signal.append(_signal)
            input_signal_length.append(_signal_len)

            if _signal_len < self.signal_chunk_len:
                input_signal[-1] = np.pad(input_signal[-1], (0, self.signal_chunk_len-_signal_len))
                break

        return input_signal, input_signal_length

    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def forward(self, input_signal, input_signal_length):
        
        all_logits = []
        for s_idx in range(0, len(input_signal), self.batch_size):
            input_signal_pt = torch.stack([torch.tensor(_, device=self.device) for _ in input_signal[s_idx:s_idx+self.batch_size]])
            input_signal_length_pt = torch.tensor(input_signal_length[s_idx:s_idx+self.batch_size], device=self.device)
            
            x, x_len = self.vad_pp(input_signal_pt, input_signal_length_pt)
            logits = self.vad_model(x, x_len)

            for _logits, _len in zip(logits, input_signal_length_pt):
                all_logits.append(_logits[:int(_len/self.signal_to_logit_len)])
        
        if len(all_logits) > 1 and self.margin_logit_len > 0:
            all_logits[0] = all_logits[0][:-self.margin_logit_len]
            all_logits[-1] = all_logits[-1][self.margin_logit_len:]

            for i in range(1, len(all_logits)-1):
                all_logits[i] = all_logits[i][self.margin_logit_len:-self.margin_logit_len]

        all_logits = torch.concatenate(all_logits)
        all_logits = torch.softmax(all_logits, dim=-1)
        
        return all_logits[:, 1].detach().cpu().numpy()
    
    def __call__(self, audio_signal):
        audio_duration = len(audio_signal)/self.sampling_rate
        
        input_signal, input_signal_length = self.prepare_input_batch(audio_signal)
        speech_probs = self.forward(input_signal, input_signal_length)

        vad_times = []
        for idx, prob in enumerate(speech_probs):
            s_time = idx*self.frame_size
            e_time = min(audio_duration, (idx+1)*self.frame_size)
            
            if s_time >= e_time: break
                
            vad_times.append([prob, s_time, e_time])

        return np.array(vad_times)
