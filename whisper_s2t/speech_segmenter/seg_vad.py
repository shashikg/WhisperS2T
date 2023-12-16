import os
import torch
import numpy as np

from . import VADBaseClass
from .. import BASE_PATH


class SegmentVAD(VADBaseClass):
    def __init__(self, 
                 device=None,
                 win_len=0.32,
                 win_step=0.08,
                 batch_size=512,
                 sampling_rate=16000):
        
        super().__init__(sampling_rate=sampling_rate)
        
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.device = device

        if self.device == 'cpu':
            # This is a JIT Scripted model of Nvidia's NeMo Marblenet Model: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/vad_multilingual_marblenet
            self.vad_pp = torch.jit.load(os.path.join(BASE_PATH, "assets/vad_pp_cpu.ts")).to(self.device)
            self.vad_model = torch.jit.load(os.path.join(BASE_PATH, "assets/seg_vad_model_cpu.ts")).to(self.device)
        else:
            self.vad_pp = torch.jit.load(os.path.join(BASE_PATH, "assets/vad_pp_gpu.ts")).to(self.device)
            self.vad_model = torch.jit.load(os.path.join(BASE_PATH, "assets/seg_vad_model_gpu.ts")).to(self.device)
        
        self.vad_pp = torch.jit.load(os.path.join(BASE_PATH, "assets/vad_pp.ts"))
        self.vad_model = torch.jit.load(os.path.join(BASE_PATH, "assets/segment_vad_model.ts"))
        
        self.vad_model.eval()
        self.vad_model.to(self.device)
        
        self.vad_pp.eval()
        self.vad_pp.to(self.device)
        
        self.batch_size = batch_size
        self.win_len = win_len
        self.win_step = win_step
        
        self._init_params()
        
    def _init_params(self):
        self.signal_win_len = int(self.win_len*self.sampling_rate)
        self.signal_win_step = int(self.win_step*self.sampling_rate)
        
    def update_params(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)
        
        self._init_params()
        
    def prepare_input_batch(self, audio_signal):

        num_chunks = (self.signal_win_len//2+len(audio_signal))//self.signal_win_step
        if num_chunks < (self.signal_win_len//2+len(audio_signal))/self.signal_win_step:
            num_chunks += 1

        input_signal = np.zeros((num_chunks, self.signal_win_len), dtype=np.float32)
        input_signal_length = np.zeros(num_chunks, dtype=np.int64)

        chunk_idx = 0
        for idx in range(-1*self.signal_win_len//2, len(audio_signal), self.signal_win_step):
            s_idx = max(idx, 0)
            e_idx = min(idx + self.signal_win_len, len(audio_signal))
            input_signal[chunk_idx][:e_idx-s_idx] = audio_signal[s_idx:e_idx]
            input_signal_length[chunk_idx] = e_idx-s_idx
            chunk_idx += 1

        return input_signal, input_signal_length
    
    @torch.cuda.amp.autocast()
    @torch.no_grad()
    def forward(self, input_signal, input_signal_length):
        x, x_len = self.vad_pp(torch.Tensor(input_signal).to(self.device), 
                               torch.Tensor(input_signal_length).to(self.device))
        logits = self.vad_model(x, x_len)
        logits = torch.softmax(logits, dim=-1)
        return logits[:, 1].detach().cpu().numpy()
    
    def __call__(self, audio_signal):
        
        audio_duration = len(audio_signal)/self.sampling_rate
        
        input_signal, input_signal_length = self.prepare_input_batch(audio_signal)
        
        speech_probs = np.zeros(len(input_signal))
        for s_idx in range(0, len(input_signal), self.batch_size):
            speech_probs[s_idx:s_idx+self.batch_size] = self.forward(input_signal=input_signal[s_idx:s_idx+self.batch_size],
                                                                     input_signal_length=input_signal_length[s_idx:s_idx+self.batch_size])

        vad_times = []
        for idx, prob in enumerate(speech_probs):
            s_time = max(0, (idx-0.5)*self.win_step)
            e_time = min(audio_duration, (idx+0.5)*self.win_step)
            vad_times.append([prob, s_time, e_time])
            
        return np.array(vad_times)
