# Adapted from NVIDIA TensorRT-LLM v1.2.1, commit 376f7e1bd8ed543f75014309e3fd4b237e9b0e73.
# Modified for WhisperS2T integration; see MODERN_TENSORRT.md and LICENSE-NVIDIA.
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
from collections import OrderedDict
from pathlib import Path
import torch
import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.llmapi.kv_cache_type import KVCacheType
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=None):
    if pad_value:
        assert (
            input_tensor_lengths is None
        ), "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert (
            input_tensor_lengths is not None
        ), "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor


def read_config(component, engine_dir):
    config_path = engine_dir / component / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config["pretrained_config"])
    model_config.update(config["build_config"])
    return model_config


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config("encoder", engine_dir)
        self.n_mels = config["n_mels"]
        self.dtype = config["dtype"]
        self.num_languages = config["num_languages"]
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / "encoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel, mel_input_lengths, encoder_downsampling_factor=2):
        if isinstance(mel, list):
            longest_mel = max([f.shape[-1] for f in mel])
            mel = [
                torch.nn.functional.pad(
                    f, (0, longest_mel - f.shape[-1]), mode="constant"
                )
                for f in mel
            ]
            mel = torch.cat(mel, dim=0).type(str_dtype_to_torch("float16")).contiguous()
        bsz, seq_len = mel.shape[0], mel.shape[2]
        position_ids = (
            torch.arange(
                math.ceil(seq_len / encoder_downsampling_factor),
                dtype=torch.int32,
                device=mel.device,
            )
            .expand(bsz, -1)
            .contiguous()
        )
        if self.encoder_config["plugin_config"]["remove_input_padding"]:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)
            position_ids = remove_tensor_padding(
                position_ids, mel_input_lengths // encoder_downsampling_factor
            )
        inputs = OrderedDict()
        inputs["input_features"] = mel
        inputs["input_lengths"] = mel_input_lengths
        inputs["position_ids"] = position_ids

        output_list = [
            TensorInfo("input_features", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo(
                "input_lengths", str_dtype_to_trt("int32"), mel_input_lengths.shape
            ),
            TensorInfo(
                "position_ids", str_dtype_to_trt("int32"), inputs["position_ids"].shape
            ),
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f"output info {output_info}")
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "Engine execution failed"
        stream.synchronize()
        encoder_output = outputs["encoder_output"]
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor
        return encoder_output, encoder_output_lengths


class WhisperDecoding:
    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):
        self.decoder_config = read_config("decoder", engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode
        )

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / "decoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_attention_heads"],
            num_kv_heads=self.decoder_config["num_attention_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            cross_attention=True,
            num_layers=self.decoder_config["num_hidden_layers"],
            gpt_attention_plugin=self.decoder_config["plugin_config"][
                "gpt_attention_plugin"
            ],
            remove_input_padding=self.decoder_config["plugin_config"][
                "remove_input_padding"
            ],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config["plugin_config"]["paged_kv_cache"]
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.decoder_config["has_position_embedding"],
            dtype=self.decoder_config["dtype"],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        encoder_max_input_length,
        encoder_input_lengths,
        eot_id,
        max_new_tokens=40,
        num_beams=1,
        sampling_options=None,
    ):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones(
                [
                    batch_size,
                    decoder_max_input_length + max_new_tokens,
                    encoder_max_input_length,
                ]
            )
            .int()
            .cuda()
        )
        # generation config
        sampling_config = sampling_options
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length,
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config["plugin_config"]["remove_input_padding"]:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID
            )
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full(
                    (encoder_outputs.shape[0],),
                    encoder_outputs.shape[1],
                    dtype=torch.int32,
                    device="cuda",
                )

                encoder_outputs = remove_tensor_padding(
                    encoder_outputs, encoder_output_lens
                )
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRT:
    def __init__(self, engine_dir, compute_type="float16"):
        engine_dir = Path(engine_dir)
        mapping = tensorrt_llm.Mapping(world_size=1, rank=0)
        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, mapping)
        self.n_mels = self.encoder.n_mels
        self.is_multilingual = self.decoder.decoder_config["vocab_size"] >= 51865
        self.compute_type = compute_type
        self.build_config = read_config("decoder", engine_dir)
        if self.build_config["plugin_config"]["remove_input_padding"]:
            raise ValueError(
                "Build with --remove_input_padding disable for this runtime"
            )

    def encode(self, mel):
        lengths = torch.full(
            (mel.shape[0],), mel.shape[-1], dtype=torch.int32, device=mel.device
        )
        return self.encoder.get_audio_features(
            mel.to(str_dtype_to_torch(self.compute_type)), lengths
        )

    def generate(self, features, prompts, **kwargs):
        if len(prompts) > self.build_config["max_batch_size"]:
            raise ValueError(
                "Batch exceeds engine capacity; rebuild or reduce batch_size"
            )
        if any(len(p) > self.build_config["max_input_len"] for p in prompts):
            raise ValueError(
                "Prompt exceeds engine capacity; rebuild with a larger max_input_len"
            )
        if len({len(p) for p in prompts}) != 1:
            raise ValueError("Static runtime requires equal prompt lengths")
        encoded, lengths = self.encode(features)
        max_new_tokens = kwargs.pop("max_new_tokens")
        config = SamplingConfig(**kwargs)
        return self.decoder.generate(
            torch.tensor(prompts, dtype=torch.int32, device=features.device),
            encoded,
            encoded.shape[1],
            lengths,
            config.end_id,
            max_new_tokens=max_new_tokens,
            num_beams=config.num_beams,
            sampling_options=config,
        )
