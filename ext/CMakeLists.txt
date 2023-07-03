# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
add_subdirectory(base)
add_subdirectory(config)
add_subdirectory(utils)
add_subdirectory(tokenizer)
add_subdirectory(model_service)
add_subdirectory(lora)

add_library(ft-ext STATIC
  $<TARGET_OBJECTS:BaseBeamSearchLayer>
  $<TARGET_OBJECTS:BaseSamplingLayer>
  $<TARGET_OBJECTS:BeamSearchLayer>
  $<TARGET_OBJECTS:Bert>
  $<TARGET_OBJECTS:BertLayerWeight>
  $<TARGET_OBJECTS:BertTritonBackend>
  $<TARGET_OBJECTS:BertWeight>
  $<TARGET_OBJECTS:DecoderCrossAttentionLayer>
  $<TARGET_OBJECTS:DecoderSelfAttentionLayer>
  $<TARGET_OBJECTS:DynamicDecodeLayer>
  $<TARGET_OBJECTS:FfnLayer>
  $<TARGET_OBJECTS:FusedAttentionLayer>
  $<TARGET_OBJECTS:GptContextAttentionLayer>
  $<TARGET_OBJECTS:GptJ>
  $<TARGET_OBJECTS:GptJContextDecoder>
  $<TARGET_OBJECTS:GptJDecoder>
  $<TARGET_OBJECTS:GptJDecoderLayerWeight>
  $<TARGET_OBJECTS:GptJTritonBackend>
  $<TARGET_OBJECTS:GptJWeight>
  $<TARGET_OBJECTS:GptNeoX>
  $<TARGET_OBJECTS:GptNeoXContextDecoder>
  $<TARGET_OBJECTS:GptNeoXDecoder>
  $<TARGET_OBJECTS:GptNeoXDecoderLayerWeight>
  $<TARGET_OBJECTS:GptNeoXTritonBackend>
  $<TARGET_OBJECTS:GptNeoXWeight>
  $<TARGET_OBJECTS:LinearAdapterLayer>
  $<TARGET_OBJECTS:OnlineBeamSearchLayer>
  $<TARGET_OBJECTS:ParallelGpt>
  $<TARGET_OBJECTS:ParallelGptContextDecoder>
  $<TARGET_OBJECTS:ParallelGptDecoder>
  $<TARGET_OBJECTS:ParallelGptDecoderLayerWeight>
  $<TARGET_OBJECTS:ParallelGptTritonBackend>
  $<TARGET_OBJECTS:ParallelGptWeight>
  $<TARGET_OBJECTS:T5Common>
  $<TARGET_OBJECTS:T5Decoder>
  $<TARGET_OBJECTS:T5Decoding>
  $<TARGET_OBJECTS:T5Encoder>
  $<TARGET_OBJECTS:T5TritonBackend>
  $<TARGET_OBJECTS:T5EncoderTritonBackend>
  $<TARGET_OBJECTS:TensorParallelDecoderCrossAttentionLayer>
  $<TARGET_OBJECTS:TensorParallelDecoderSelfAttentionLayer>
  $<TARGET_OBJECTS:TensorParallelGeluFfnLayer>
  $<TARGET_OBJECTS:TensorParallelSiluFfnLayer>
  $<TARGET_OBJECTS:TensorParallelGptContextAttentionLayer>
  $<TARGET_OBJECTS:TensorParallelReluFfnLayer>
  $<TARGET_OBJECTS:TensorParallelUnfusedAttentionLayer>
  $<TARGET_OBJECTS:TopKSamplingLayer>
  $<TARGET_OBJECTS:TopPSamplingLayer>
  $<TARGET_OBJECTS:TransformerTritonBackend>
  $<TARGET_OBJECTS:UnfusedAttentionLayer>
  $<TARGET_OBJECTS:activation_int8_kernels>
  $<TARGET_OBJECTS:activation_kernels>
  $<TARGET_OBJECTS:add_bias_transpose_kernels>
  $<TARGET_OBJECTS:add_residual_kernels>
  $<TARGET_OBJECTS:ban_bad_words>
  $<TARGET_OBJECTS:beam_search_penalty_kernels>
  $<TARGET_OBJECTS:beam_search_topk_kernels>
  $<TARGET_OBJECTS:bert_preprocess_kernels>
  $<TARGET_OBJECTS:calibrate_quantize_weight_kernels>
  $<TARGET_OBJECTS:cublasAlgoMap>
  $<TARGET_OBJECTS:cublasMMWrapper>
  $<TARGET_OBJECTS:cuda_driver_wrapper>
  $<TARGET_OBJECTS:cuda_utils>
  $<TARGET_OBJECTS:custom_ar_comm>
  $<TARGET_OBJECTS:custom_ar_kernels>
  $<TARGET_OBJECTS:cutlass_heuristic>
  $<TARGET_OBJECTS:cutlass_preprocessors>
  $<TARGET_OBJECTS:decoder_masked_multihead_attention>
  $<TARGET_OBJECTS:decoding_kernels>
  $<TARGET_OBJECTS:fpA_intB_gemm>
  $<TARGET_OBJECTS:gen_relative_pos_bias>
  $<TARGET_OBJECTS:gpt_kernels>
  $<TARGET_OBJECTS:int8_gemm>
  $<TARGET_OBJECTS:layernorm_int8_kernels>
  $<TARGET_OBJECTS:layernorm_kernels>
  $<TARGET_OBJECTS:layout_transformer_int8_kernels>
  $<TARGET_OBJECTS:logprob_kernels>
  $<TARGET_OBJECTS:logger>
  $<TARGET_OBJECTS:longformer_kernels>
  $<TARGET_OBJECTS:matrix_transpose_kernels>
  $<TARGET_OBJECTS:matrix_vector_multiplication>
  $<TARGET_OBJECTS:memory_utils>
  $<TARGET_OBJECTS:moe_gemm_kernels>
  $<TARGET_OBJECTS:moe_kernels>
  $<TARGET_OBJECTS:mpi_utils>
  $<TARGET_OBJECTS:nccl_utils>
  $<TARGET_OBJECTS:nvtx_utils>
  $<TARGET_OBJECTS:online_softmax_beamsearch_kernels>
  $<TARGET_OBJECTS:quantization_int8_kernels>
  $<TARGET_OBJECTS:sampling_penalty_kernels>
  $<TARGET_OBJECTS:sampling_topk_kernels>
  $<TARGET_OBJECTS:sampling_topp_kernels>
  $<TARGET_OBJECTS:softmax_int8_kernels>
  $<TARGET_OBJECTS:stop_criteria>
  $<TARGET_OBJECTS:tensor>
  $<TARGET_OBJECTS:transpose_int8_kernels>
  $<TARGET_OBJECTS:trt_fused_multi_head_attention>
  $<TARGET_OBJECTS:unfused_attention_kernels>
  $<TARGET_OBJECTS:word_list>
  $<TARGET_OBJECTS:ustring>
  $<TARGET_OBJECTS:model_config>
  $<TARGET_OBJECTS:model_service_instance>
  $<TARGET_OBJECTS:model_service>
  $<TARGET_OBJECTS:basic_tokenizer>
  $<TARGET_OBJECTS:bert_tokenizer>
  $<TARGET_OBJECTS:bert_tokenizer_decoder>
  $<TARGET_OBJECTS:time_utils>
  $<TARGET_OBJECTS:file_utils>
  $<TARGET_OBJECTS:string_utils>
  $<TARGET_OBJECTS:spdlogger>
)



