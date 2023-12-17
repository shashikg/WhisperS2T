# Detailed Usage and Documentation

## Basic Usage

Load WhisperS2T with CTranslate2 backend with default parameters:

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v2", backend='CTranslate2')

files = ['sample_1.wav']
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes, # pass lang_codes for each file
                                tasks=tasks, # pass transcribe/translate 
                                initial_prompts=initial_prompts, # to do prompting (currently only supported for CTranslate2 backend)
                                batch_size=16)

```

Switch to HuggingFace backend (by default it will use FlashAttention2). Note: FlashAttention2 only works with Ampere/Hopper Nvidia GPUs.

```py
model = whisper_s2t.load_model(model_identifier="large-v2", backend='HuggingFace') # Supported backends ['CTranslate2', 'HuggingFace', 'OpenAI']
```

## Using Custom VAD Model

Wrap your VAD model (say `CustomVAD`) using the base class as `whisper_s2t.speech_segmenter.VADBaseClass`. See [whisper_s2t/speech_segmenter/frame_vad.py](whisper_s2t/speech_segmenter/frame_vad.py) for example. The `def __call__` must take audio_signal as input and returns a numpy array of size **T x 3** where T is the frame length. Each row should have following data `[speech_prob, frame_start_time, frame_end_time]`. Next pass your vad_model while initialising the whisper model.

```py
vad_model = CustomVAD()
model = whisper_s2t.load_model(model_identifier="large-v2", backend='CTranslate2', vad_model=vad_model)
```

## Passing Custom Model Configuration

Custom model configs can be passed as keyword arguments when loading the model:

```py
import whisper_s2t
from whisper_s2t.backends.ctranslate2.model import BEST_ASR_CONFIG

model_kwargs = {
    'compute_type': 'int8', # Note int8 is only supported for CTranslate2 backend, for others only float16 is supported for lower precision.
    'asr_options': BEST_ASR_CONFIG
}

model = whisper_s2t.load_model(model_identifier="large-v2", backend='CTranslate2', **model_kwargs)
```

OR to update the configs after loading the model:

```py
model.update_params(model_kwargs)
```