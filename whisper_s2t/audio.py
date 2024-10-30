import io
import os
import wave
import tempfile
import subprocess
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import concurrent
# from multiprocessing.dummy import Pool

from . import BASE_PATH
from .configs import *

silent_file = f"{BASE_PATH}/assets/silent.mp3"

RESAMPLING_ENGINE = 'soxr'
with tempfile.TemporaryDirectory() as tmpdir:
    ffmpeg_install_link = "https://github.com/shashikg/WhisperS2T?tab=readme-ov-file#for-ubuntu"
    
    try: 
        subprocess.check_output(['ffmpeg', '-version'])
    except:
        raise RuntimeError(f"Seems 'ffmpeg' is not installed. Please install ffmpeg before using this package!\nCheck: {ffmpeg_install_link}")

    ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i "{silent_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar 1600 "{tmpdir}/tmp.wav" -y')

    if ret_code != 0:
        print(f"'ffmpeg' failed with soxr resampler, trying 'swr' resampler.")
        RESAMPLING_ENGINE = 'swr'

        ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i "{silent_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar 1600 "{tmpdir}/tmp.wav" -y')

        if ret_code != 0:
            raise RuntimeError(f"Seems 'ffmpeg' is not installed properly. Please uninstall and install it again.\nCheck: {ffmpeg_install_link}")
        else:
            print(f"Using 'swr' resampler. This may degrade performance.")
        

# def load_audio(input_file, sr=16000, return_duration=False):
    
#     try:
#         with wave.open(input_file, 'rb') as wf:
#             if (wf.getframerate() != sr) or (wf.getnchannels() != 1):
#                 raise Exception("Not a 16kHz wav mono channel file!")
                
#             frames = wf.getnframes()
#             x = wf.readframes(int(frames))
#     except:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             wav_file = f"{tmpdir}/tmp.wav"
#             ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i "{input_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar {sr} "{wav_file}" -y')
#             if ret_code != 0: raise RuntimeError("ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!")
        
#             with wave.open(wav_file, 'rb') as wf:
#                 frames = wf.getnframes()
#                 x = wf.readframes(int(frames))
    
#     audio_signal = np.frombuffer(x, np.int16).flatten().astype(np.float32)/32768.0
#     audio_duration = len(audio_signal)/sr
    
#     if return_duration:
#         return audio_signal, audio_duration
#     else:
#         return audio_signal


def load_audio(input_file: str | bytes | np.ndarray, sr: int = 16000, return_duration: bool = False) -> np.ndarray | tuple[np.ndarray, float]:
    """Load audio from disk or memory

    Args:
        input_file (str | bytes | np.ndarray): path to file, audio object in memory or numpy pre-loaded ndarray
        sr (int, optional): sample rate. Defaults to 16000.
        return_duration (bool, optional): return audio duration. Defaults to False.
    
    Returns:
        (np.ndarray | tuple[np.ndarray, float]): audio signal as numpy ndarray, audio duration
    """
    def _load_audio_as_ndarray(input_file: str | bytes, sr: int = 16000) -> tuple[np.ndarray, float]:
        """Load audio from WAV file

        Args:
            input_file (str | bytes): path to file or object in memory
            sr (int, optional): sample rate. Defaults to 16000.

        Raises:
            Exception: Not a 16kHz wav mono channel file!

        Returns:
            tuple[np.ndarray, float]: audio signal as numpy ndarray, audio duration
        """
        with wave.open(input_file if isinstance(input_file, str) else io.BytesIO(input_file), 'rb') as wf:
            if (wf.getframerate() != sr) or (wf.getnchannels() != 1):
                raise Exception("Not a 16kHz wav mono channel file!")
            
            frames = wf.getnframes()
            x = wf.readframes(int(frames))
        
        # convert to numpy and calculate audio duration
        audio_signal = np.frombuffer(x, np.int16).flatten().astype(np.float32)/32768.0
        audio_duration = len(audio_signal)/sr
        
        return audio_signal, audio_duration
    
    def _ffmpeg_convert_to_wav(input_file: str, wav_file: str, sr: int = 16000):
        """Converts audio file into WAV file format

        Args:
            input_file (str): input file
            wav_file (str): wav file name
            sr (int, optional): sample rate. Defaults to 16000.

        Raises:
            RuntimeError: ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!
        """
        ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i "{input_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar {sr} "{wav_file}" -y')
        if ret_code != 0: raise RuntimeError("ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!")
    
    # load audio from disk or memory
    audio_signal = None
    audio_duration = None
    if isinstance(input_file, (str, bytes)):
        try:
            audio_signal, audio_duration = _load_audio_as_ndarray(input_file=input_file, sr=sr)
        except:
            with tempfile.TemporaryDirectory() as tmpdir:
                # save bytes to file
                if isinstance(input_file, bytes):
                    tmp_file = os.path.join(tmpdir, 'audio')
                    with open(tmp_file, 'wb') as f:
                        f.write(input_file)
                    input_file = tmp_file
                
                # convert to wav
                wav_file = os.path.join(tmpdir, 'tmp.wav')
                _ffmpeg_convert_to_wav(input_file=input_file, wav_file=wav_file, sr=sr)
                audio_signal, audio_duration = _load_audio_as_ndarray(input_file=wav_file, sr=sr)
        
    # already preprocessed into numpy ndarray
    else: 
        audio_signal = input_file
        audio_duration = len(input_file) / sr
    
    return (audio_signal, audio_duration) if return_duration else audio_signal


# THREAD_POOL_AUDIO_LOADER = Pool(2)
# def audio_batch_generator(audio_files):
#     return THREAD_POOL_AUDIO_LOADER.imap(load_audio, audio_files)


def audio_batch_generator(audio_files: list, parallel: bool = True, max_workers: int = 2):
    """
    Generate batches of loaded audio files, with option for parallel or sequential loading.
    
    Args:
        audio_files (list): list of paths to audio files
        parallel (bool, optional, default=True): tries parallel loading if True else uses sequential loading
        max_workers (int, optional, default=2): maximum number of parallel workers (only used if parallel=True)
    
    Returns:
        Iterator of loaded audio data
    """
    # try parallel loading with ThreadPoolExecutor (safer than multiprocessing.dummy.Pool)
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try: 
                yield from executor.map(load_audio, audio_files)
                return # if parallel loading succeeded, we are done
            except Exception as e:
                print(f'Parallel audio loading failed: {str(e)}. Fall back to sequential loading...')
                parallel = False            
    
    # sequential loading (fallback)
    for audio_file in audio_files:
        yield load_audio(audio_file)


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