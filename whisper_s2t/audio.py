import os
import wave
import tempfile
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing.dummy import Pool

from . import BASE_PATH
from .configs import *

silent_file = f"{BASE_PATH}/assets/silent.mp3"

RESAMPLING_ENGINE = 'soxr'
with tempfile.TemporaryDirectory() as tmpdir:
    ret_code = os.system(f'ffmpeg -version')
    if ret_code != 0:
        print(f"Seems 'ffmpeg' is not installed. Please install ffmpeg before using this package!")
    else:
        ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i {silent_file} -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar 1600 {tmpdir}/tmp.wav -y')

        if ret_code != 0:
            print(f"'ffmpeg' is not built with soxr resampler, using 'swr' resampler. This may degrade performance.")
            RESAMPLING_ENGINE = 'swr'

def load_audio(input_file, sr=16000, return_duration=False):
    
    try:
        with wave.open(input_file, 'rb') as wf:
            if (wf.getframerate() != sr) or (wf.getnchannels() != 1):
                raise Exception("Not a 16kHz wav mono channel file!")
                
            frames = wf.getnframes()
            x = wf.readframes(int(frames))
    except:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_file = f"{tmpdir}/tmp.wav"
            ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i {input_file} -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar {sr} {wav_file} -y')
            if ret_code != 0: raise RuntimeError("ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!")
        
            with wave.open(wav_file, 'rb') as wf:
                frames = wf.getnframes()
                x = wf.readframes(int(frames))
    
    audio_signal = np.frombuffer(x, np.int16).flatten().astype(np.float32)/32768.0
    audio_duration = len(audio_signal)/sr
    
    if return_duration:
        return audio_signal, audio_duration
    else:
        return audio_signal


THREAD_POOL_AUDIO_LOADER = Pool(2)
def audio_batch_generator(audio_files):
    return THREAD_POOL_AUDIO_LOADER.imap(load_audio, audio_files)


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)
    
    return array


class TorchSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        
    def forward(self, x):
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True)


class LogMelSpectogram(nn.Module):
    def __init__(self, 
                 n_mels=N_MELS,
                 n_fft=N_FFT,
                 hop_length=HOP_LENGTH,
                 padding=0):
        
        super().__init__()
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.padding = padding
        
        mel_filters = np.load(os.path.join(BASE_PATH, "assets/mel_filters.npz"))
        mel_filters = torch.from_numpy(mel_filters[f"mel_{n_mels}"])
        self.register_buffer("mel_filters", mel_filters)
        
        self.stft = TorchSTFT(n_fft, hop_length)
        
    def get_seq_len(self, seq_len):
        seq_len = torch.floor(seq_len/self.hop_length)
        return seq_len.to(dtype=torch.long)
    
    @torch.no_grad()
    def forward(self, x, seq_len):
        
        seq_len = self.get_seq_len(seq_len.float())
        
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))
            
        x = self.stft(x)
        
        x = x[..., :-1].abs()**2
        x = self.mel_filters@x # mels

        x = torch.clamp(x, min=1e-10).log10() # log_mels
        x = torch.maximum(x, torch.amax(x, dim=(1, 2), keepdims=True) - 8.0) # clip
        x = (x + 4.0) / 4.0 # scale
        
        return x, seq_len