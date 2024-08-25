import os
import wave
import tempfile
import subprocess

import numpy as np
from whisper_s2t import BASE_PATH

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
        

def convert_to_wav16(input_file, output_file, sr=16000):
    ret_code = os.system(f'ffmpeg -hide_banner -loglevel panic -i "{input_file}" -threads 1 -acodec pcm_s16le -ac 1 -af aresample=resampler={RESAMPLING_ENGINE} -ar {sr} "{output_file}" -y')
    if ret_code != 0: raise RuntimeError("ffmpeg failed to resample the input audio file, make sure ffmpeg is compiled properly!")


def load_audio(input_file, sr=16000, return_duration=True):
    
    with wave.open(input_file, 'rb') as wf:
        if (wf.getframerate() != sr) or (wf.getnchannels() != 1):
            raise Exception("Not a 16kHz wav mono channel file!")
            
        frames = wf.getnframes()
        x = wf.readframes(int(frames))
    
    audio_signal = np.frombuffer(x, np.int16).flatten().astype(np.float32)/32768.0
    audio_duration = len(audio_signal)/sr
    
    if return_duration:
        return audio_signal, audio_duration
    else:
        return audio_signal

def stitch_speech_segments(start_ends, max_len=27.0, max_silent_region=None):

    speech_duration = [end - start for start, end in start_ends]
    
    stitched_speech_segments = []
    
    curr_seg = [0]
    curr_dur = speech_duration[0]
    idx = 1
    while idx < len(start_ends):
        if curr_dur + speech_duration[idx] > max_len:
            stitched_speech_segments.append([start_ends[_] for _ in curr_seg])
            curr_seg = [idx]
            curr_dur = speech_duration[idx]
        else:
            curr_dur += speech_duration[idx]
            curr_seg.append(idx)
            
        idx += 1
        
    stitched_speech_segments.append([start_ends[_] for _ in curr_seg])
    
    if max_silent_region is None:
        return stitched_speech_segments
    
    stitched_speech_segments_joined = []
    for segs in stitched_speech_segments:
        _segs = []
        curr_seg_start_time, curr_seg_end_time = segs[0]
        for i in range(1, len(segs)):
            if (segs[i][0] - curr_seg_end_time) >= max_silent_region:
                _segs.append((curr_seg_start_time, curr_seg_end_time))
                curr_seg_start_time = segs[i][0]

            curr_seg_end_time = segs[i][1]

        _segs.append((curr_seg_start_time, curr_seg_end_time))
        
        stitched_speech_segments_joined.append(_segs)
        
    
    return stitched_speech_segments_joined


MAX_SPEECH_LEN = float(os.getenv("MAX_SPEECH_LEN", "29.0"))
MERGE_CHUNKS = False if os.getenv("MERGE_CHUNKS", "True") == "False" else True

def get_segmented_audio_signal(start_ends, audio_signal, sr=16000):

    segmented_audio_signal = []

    if MERGE_CHUNKS:
        stitched_speech_segments = stitch_speech_segments(start_ends, max_len=MAX_SPEECH_LEN)
        for stitched_seg in stitched_speech_segments:
            audio = []
            for st, et in stitched_seg:
                audio.append(audio_signal[int(st*sr):int(et*sr)])

            audio = np.concatenate(audio)
            seq_len = audio.shape[-1]
            seg_metadata = {
                'start_time': stitched_seg[0][0], 
                'end_time': stitched_seg[-1][1], 
                'stitched_seg': stitched_seg,
            }
            segmented_audio_signal.append((audio, seq_len, seg_metadata))
    else:
        for st, et in start_ends:
            audio = audio_signal[int(st*sr):int(et*sr)]
            seq_len = audio.shape[-1]
            segmented_audio_signal.append((audio, seq_len, {'start_time': st, 'end_time': et}))

    return segmented_audio_signal