# Detailed Usage and Documentation

1. [Basic Usage](#basic-usage)
1. [Using Custom VAD Model](#using-custom-vad-model)
1. [Run Without VAD Model](#run-without-vad-model)
1. [Passing Custom Model Configuration](#passing-custom-model-configuration)
1. [Return Word-Alignments](#return-word-alignments)
1. [Write Transcripts To a File](#write-transcripts-to-a-file)

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

print(out[0][0]) # Print first utterance for first file
"""
[Console Output]

{'text': "Let's bring in Phil Mackie who is there at the palace. We're looking at Teresa and Philip May. Philip, can you see how he's being transferred from the helicopters? It looks like, as you said, the beast. It's got its headlights on because the sun is beginning to set now, certainly sinking behind some clouds. It's about a quarter of a mile away down the Grand Drive",
 'avg_logprob': -0.25426941679184695,
 'no_speech_prob': 8.147954940795898e-05,
 'start_time': 0.0,
 'end_time': 24.8}
"""

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

## Run Without VAD Model

For some languages VAD model can give poor performance. For those cases, it's better to disable VAD.

```py
out = model.transcribe(files,
                       lang_codes=lang_codes, # pass lang_codes for each file
                       tasks=tasks, # pass transcribe/translate 
                       initial_prompts=initial_prompts, # to do prompting (currently only supported for CTranslate2 backend)
                       batch_size=24)

print(out[0][0])
"""
{'text': "Let's bring in Phil Mackie who is there at the palace. We're looking at Theresa and Philip May. Philip, can you see how he's being transferred from the helicopters? It looks like, as you said, the beast. It's got its headlights on because the sun is beginning to set now, certainly sinking behind some clouds. It's about a quarter of a mile away down the Grand Drive leading up into the courtyard. So you've seen the pictures there of the Prime Minister",
 'avg_logprob': -0.25300603330135346,
 'no_speech_prob': 1.9311904907226562e-05,
 'start_time': 0,
 'end_time': 29.0}
"""
```

VAD parameters can also be tweaked using:

```py
speech_segmenter_options = {
    'eos_thresh': 0.1,
    'bos_thresh': 0.1,
}

model = whisper_s2t.load_model(speech_segmenter_options=speech_segmenter_options)
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

## Return Word-Alignments

Only for CTranslate2 and TensorRT backend.

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v2", asr_options={'word_timestamps': True})

files = ['sample_1.wav']
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes, # pass lang_codes for each file
                                tasks=tasks, # pass transcribe/translate 
                                initial_prompts=initial_prompts, # to do prompting (currently only supported for CTranslate2 backend)
                                batch_size=24)

print(out[0][0]) # Print first utterance for first file
"""
[Console Output]

{'text': "Let's bring in Phil Mackie who is there at the palace. We're looking at Teresa and Philip May. Philip, can you see how he's being transferred from the helicopters? It looks like, as you said, the beast. It's got its headlights on because the sun is beginning to set now, certainly sinking behind some clouds. It's about a quarter of a mile away down the Grand Drive",
 'avg_logprob': -0.2544597674565143,
 'no_speech_prob': 8.213520050048828e-05,
 'word_timestamps': [{'word': "Let's",
   'start': 0.0,
   'end': 0.24,
   'prob': 0.63},
  {'word': 'bring', 'start': 0.24, 'end': 0.4, 'prob': 0.96},
  {'word': 'in', 'start': 0.4, 'end': 0.52, 'prob': 0.71},
  {'word': 'Phil', 'start': 0.52, 'end': 0.66, 'prob': 0.46},
  {'word': 'Mackie', 'start': 0.66, 'end': 1.02, 'prob': 0.27},
  {'word': 'who', 'start': 1.02, 'end': 1.2, 'prob': 0.01},
  .
  .
  .
  .
}
"""
```

## Write Transcripts To a File

Predicted transcripts can be easily exported to following output formats: `vtt, srt, json, tsv`.

```py
files = ['file.wav']
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

out = model.transcribe_with_vad(files,
                                lang_codes=lang_codes,
                                tasks=tasks,
                                initial_prompts=initial_prompts,
                                batch_size=24)

whisper_s2t.write_outputs(out, format='vtt', ip_files=files, save_dir="./save_dir") # Save outputs

whisper_s2t.write_outputs(out, format='vtt', op_files=op_files) # custom output file names
```