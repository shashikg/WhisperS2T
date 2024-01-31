import json
import torch
import tensorrt_llm

from pathlib import Path
from collections import OrderedDict

from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / 'encoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        dtype = config['builder_config']['precision']
        n_mels = config['builder_config']['n_mels']
        num_languages = config['builder_config']['num_languages']

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f'encoder.engine'

        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)

        inputs = OrderedDict()
        inputs['x'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features
        

class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_config(self, engine_dir):
        config_path = engine_dir / 'decoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config['plugin_config'])
        decoder_config.update(config['builder_config'])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        dtype = self.decoder_config['precision']
        serialize_path = engine_dir / f'decoder.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            num_heads=self.decoder_config['num_heads'],
            num_kv_heads=self.decoder_config['num_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            num_layers=self.decoder_config['num_layers'],
            gpt_attention_plugin=self.decoder_config['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['remove_input_padding'],
            cross_attention=self.decoder_config['cross_attention'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            has_token_type_embedding=self.
            decoder_config['has_token_type_embedding'],
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 sampling_config):
        
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device='cuda')

        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            encoder_max_input_length=encoder_outputs.shape[1])

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRT:
    def __init__(self, engine_dir, compute_type='float16'):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        
        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, runtime_mapping)
        self.n_mels = self.encoder.n_mels
        self.is_multilingual = True
        self.compute_type = compute_type
        
    def encode(self, mel):
        return self.encoder.get_audio_features(mel.type(str_dtype_to_torch(self.compute_type)))

    def generate(self, features, prompts, **generate_kwargs):
        if features.shape[1] == self.n_mels:
            features = self.encode(features)

        decoder_input_ids = torch.tensor(prompts)
            
        sampling_config = SamplingConfig(**generate_kwargs)
        
        output_ids = self.decoder.generate(decoder_input_ids,
                                           features,
                                           sampling_config)

        return output_ids