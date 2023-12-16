import numpy as np
from abc import ABC, abstractmethod

from ..audio import load_audio


class VADBaseClass(ABC):
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        
    @abstractmethod
    def update_params(self, params={}):
        pass
    
    @abstractmethod
    def __call__(self, audio_signal, batch_size=4):
        pass
    

class SpeechSegmenter:
    def __init__(self, vad_model=None,
                 device=None,
                 frame_size=0.02,
                 min_seg_len=0.08,
                 max_seg_len=29.0,
                 max_silent_region=0.6,
                 padding=0.2,
                 eos_thresh=0.3,
                 bos_thresh=0.3,
                 cut_factor=2,
                 sampling_rate=16000):
        
        if vad_model is None:
            from .frame_vad import FrameVAD
            vad_model = FrameVAD(device=device)
        
        self.vad_model = vad_model
        
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.frame_size = frame_size
        self.min_seg_len = min_seg_len
        self.max_seg_len = max_seg_len
        self.max_silent_region = max_silent_region
        
        self.eos_thresh = eos_thresh
        self.bos_thresh = bos_thresh
        
        self.cut_factor = cut_factor
        self.cut_idx = int(self.max_seg_len/(self.cut_factor*self.frame_size))
        self.max_idx_in_seg = self.cut_factor*self.cut_idx
        
    def update_params(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)
        
        self.cut_idx = int(self.max_seg_len/(self.cut_factor*self.frame_size))
        self.max_idx_in_seg = self.cut_factor*self.cut_idx
        
    def update_vad_model_params(self, params={}):
        self.vad_model.update_params(params=params)
        
    def okay_to_merge(self, speech_probs, last_seg, curr_seg):
        conditions = [
            (speech_probs[curr_seg['start']][1]-speech_probs[last_seg['end']][2]) < self.max_silent_region,
            (speech_probs[curr_seg['end']][2]-speech_probs[last_seg['start']][1]) <= self.max_seg_len, 
        ]

        return all(conditions)
        
    def get_speech_segments(self, speech_probs):
        
        speech_flag, start_idx = False, 0
        speech_segments = []
        for idx, (speech_prob, st, et) in enumerate(speech_probs):
            if speech_flag:
                if speech_prob < self.eos_thresh:
                    speech_flag = False
                    curr_seg = {'start': start_idx, 'end': idx-1}

                    if len(speech_segments) and self.okay_to_merge(speech_probs, speech_segments[-1], curr_seg):
                        speech_segments[-1]['end'] = curr_seg['end']
                    else:
                        speech_segments.append(curr_seg)

            elif speech_prob >= self.bos_thresh:
                speech_flag = True
                start_idx = idx

        if speech_flag:
            curr_seg = {'start': start_idx, 'end': len(speech_probs)-1}

            if len(speech_segments) and self.okay_to_merge(speech_probs, speech_segments[-1], curr_seg):
                speech_segments[-1]['end'] = curr_seg['end']
            else:
                speech_segments.append(curr_seg)
        
        speech_segments = [_ for _ in speech_segments if (speech_probs[_['end']][2]-speech_probs[_['start']][1]) > self.min_seg_len]

        start_ends = []
        for _ in speech_segments:
            first_idx = len(start_ends)
            start_idx, end_idx = _['start'], _['end']
            while (end_idx-start_idx) > self.max_idx_in_seg:
                _start_idx = int(start_idx + self.cut_idx)
                _end_idx = int(min(end_idx, start_idx + self.max_idx_in_seg))

                new_end_idx = _start_idx+np.argmin(speech_probs[_start_idx:_end_idx, 0])
                start_ends.append([speech_probs[start_idx][1], speech_probs[new_end_idx][2]])
                start_idx = new_end_idx+1
            
            start_ends.append([speech_probs[start_idx][1], speech_probs[end_idx][2]+self.padding])
            start_ends[first_idx][0] = start_ends[first_idx][0]-self.padding
        
        return start_ends
             
    
    def __call__(self, input_file=None, audio_signal=None):
        if audio_signal is None:
            audio_signal, audio_duration = load_audio(input_file, sr=self.sampling_rate, return_duration=True)
        else:
            audio_duration = len(audio_signal)/self.sampling_rate
            
        speech_probs = self.vad_model(audio_signal)
        start_ends = self.get_speech_segments(speech_probs)
        
        start_ends[0][0] = max(0.0, start_ends[0][0]) # fix edges
        start_ends[-1][1] = min(audio_duration, start_ends[-1][1]) # fix edges
        
        return start_ends, audio_signal