# Modified from: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper

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

import os
import time
import argparse

import torch
import tensorrt_llm
from tensorrt_llm import str_dtype_to_torch, str_dtype_to_trt

from tensorrt_llm.logger import logger
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.models import quantize_model
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType

from . import load_trt_build_config
from .model_utils import load_encoder_weight, load_decoder_weight


def get_export_size(output_path):
    return os.popen(f'du -sh {output_path}').read().split("\t")[0]


def serialize_engine(engine, path):
    with open(path, 'wb') as f:
        f.write(engine)
        

def build_encoder(model, args):
    
    model_metadata = model['dims']
    model_params = model['model_state_dict']

    # cast params according dtype
    for k, v in model_params.items():
        model_params[k] = v.to(str_dtype_to_torch(args.dtype))

    builder = Builder()
    
    max_batch_size = args.max_batch_size
    hidden_states = model_metadata['n_audio_state']
    num_heads = model_metadata['n_audio_head']
    num_layers = model_metadata['n_audio_layer']

    model_is_multilingual = (model_metadata['n_vocab'] >= 51865)

    builder_config = builder.create_builder_config(
        name="encoder",
        precision=args.dtype,
        tensor_parallel=1,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_states,
        max_batch_size=max_batch_size,
        int8=args.quant_mode_enc.has_act_or_weight_quant(),
        n_mels=model_metadata['n_mels'],
        num_languages=model_metadata['n_vocab'] - 51765 -
        int(model_is_multilingual),
    )

    tensorrt_llm_whisper_encoder = tensorrt_llm.models.WhisperEncoder(
        model_metadata['n_mels'], model_metadata['n_audio_ctx'],
        model_metadata['n_audio_state'], model_metadata['n_audio_head'],
        model_metadata['n_audio_layer'], str_dtype_to_trt(args.dtype))
    
    
    if args.use_weight_only_enc:
        tensorrt_llm_whisper_encoder = quantize_model(
            tensorrt_llm_whisper_encoder, args.quant_mode_enc)

    load_encoder_weight(tensorrt_llm_whisper_encoder, model_metadata,
                        model_params, model_metadata['n_audio_layer'])

    network = builder.create_network()

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(dtype=args.use_layernorm_plugin)
        
    if args.use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(dtype=args.use_bert_attention_plugin)

        if args.use_context_fmha_enc:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    
    if args.use_weight_only_enc:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)
    
    with net_guard(network):
        inputs = tensorrt_llm_whisper_encoder.prepare_inputs(
            args.max_batch_size)

        tensorrt_llm_whisper_encoder(*inputs)

        if args.debug_mode:
            for k, v in tensorrt_llm_whisper_encoder.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(args.dtype))

    engine = None
    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'encoder_config.json')
    builder.save_config(builder_config, config_path)
    
    serialize_engine(engine, os.path.join(args.output_dir, "encoder.engine"))


def build_decoder(model, args):

    model_metadata = model['dims']
    model_params = model['model_state_dict']

    # cast params according dtype
    for k, v in model_params.items():
        model_params[k] = v.to(str_dtype_to_torch(args.dtype))

    builder = Builder()

    timing_cache_file = os.path.join(args.output_dir, 'decoder_model.cache')
    builder_config = builder.create_builder_config(
        name="decoder",
        precision=args.dtype,
        timing_cache=timing_cache_file,
        tensor_parallel=args.world_size,
        num_layers=model_metadata['n_text_layer'],
        num_heads=model_metadata['n_text_head'],
        hidden_size=model_metadata['n_text_state'],
        vocab_size=model_metadata['n_vocab'],
        hidden_act="gelu",
        max_position_embeddings=model_metadata['n_text_ctx'],
        apply_query_key_layer_scaling=False,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        opt_level=None,
        cross_attention=True,
        has_position_embedding=True,
        has_token_type_embedding=False,
        int8=args.quant_mode_dec.has_act_or_weight_quant()
    )

    tensorrt_llm_whisper_decoder = tensorrt_llm.models.DecoderModel(
        num_layers=model_metadata['n_text_layer'],
        num_heads=model_metadata['n_text_head'],
        hidden_size=model_metadata['n_text_state'],
        ffn_hidden_size=4 * model_metadata['n_text_state'],
        encoder_hidden_size=model_metadata['n_text_state'],
        encoder_num_heads=model_metadata['n_text_head'],
        vocab_size=model_metadata['n_vocab'],
        head_size=model_metadata['n_text_state'] //
        model_metadata['n_text_head'],
        max_position_embeddings=model_metadata['n_text_ctx'],
        has_position_embedding=True,
        relative_attention=False,
        max_distance=0,
        num_buckets=0,
        has_embedding_layernorm=False,
        has_embedding_scale=False,
        q_scaling=1.0,
        has_attention_qkvo_bias=True,
        has_mlp_bias=True,
        has_model_final_layernorm=True,
        layernorm_eps=1e-5,
        layernorm_position=LayerNormPositionType.pre_layernorm,
        layernorm_type=LayerNormType.LayerNorm,
        hidden_act="gelu",
        rescale_before_lm_head=False,
        dtype=str_dtype_to_trt(args.dtype),
        logits_dtype=str_dtype_to_trt(args.dtype))

    if args.use_weight_only_dec:
        tensorrt_llm_whisper_decoder = quantize_model(
            tensorrt_llm_whisper_decoder, args.quant_mode_dec)

    load_decoder_weight(
        tensorrt_llm_whisper_decoder,
        model_params,
    )

    network = builder.create_network()

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(dtype=args.use_layernorm_plugin)
        
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(dtype=args.use_gpt_attention_plugin)

        if args.use_context_fmha_dec:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    with net_guard(network):
        inputs = tensorrt_llm_whisper_decoder.prepare_inputs(
            args.max_batch_size,
            args.max_beam_width,
            args.max_input_len,
            args.max_output_len,
            model_metadata['n_audio_ctx'],
        )

        tensorrt_llm_whisper_decoder(*inputs)

        if args.debug_mode:
            for k, v in tensorrt_llm_whisper_decoder.named_network_outputs():
                network._mark_output(v, k, str_dtype_to_trt(args.dtype))

    engine = None
    engine = builder.build_engine(network, builder_config)

    config_path = os.path.join(args.output_dir, 'decoder_config.json')
    builder.save_config(builder_config, config_path)

    serialize_engine(engine, os.path.join(args.output_dir, "decoder.engine"))


def run(args=None, log_level='error'):

    logger.set_level(log_level)
        
    if args.use_weight_only_enc:
        args.quant_mode_enc = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            use_int4_weights="int4" in args.weight_only_precision)
    else:
        args.quant_mode_enc = QuantMode(0)

    if args.use_weight_only_dec:
        args.quant_mode_dec = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            use_int4_weights="int4" in args.weight_only_precision)
    else:
        args.quant_mode_dec = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode_dec = args.quant_mode.set_int8_kv_cache()

    model = torch.load(args.model_path)

    _t = time.time()
    build_encoder(model, args)    
    _te = time.time()-_t

    _t = time.time()
    build_decoder(model, args)
    _td = time.time()-_t

    print(f"Time taken for building Encoder: {_te:.2f} seconds.")
    print(f"Time taken for building Decoder: {_td:.2f} seconds.")
    print(f"Exported model size: {get_export_size(args.output_dir)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--log_level', type=str)
    args = parser.parse_args()

    trt_build_args = load_trt_build_config(args.output_dir)

    print(f"[TRTBuilderConfig]:")
    print(vars(trt_build_args))
    
    run(args=trt_build_args, log_level=args.log_level)