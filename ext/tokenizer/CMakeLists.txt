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

cmake_minimum_required(VERSION 3.8)


add_library(basic_tokenizer STATIC basic_tokenizer.cc)
set_property(TARGET basic_tokenizer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET basic_tokenizer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(basic_tokenizer PUBLIC ustring string_utils)


add_library(bert_tokenizer STATIC bert_tokenizer.cc)
set_property(TARGET bert_tokenizer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET bert_tokenizer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(bert_tokenizer PUBLIC basic_tokenizer)


add_library(bert_tokenizer_decoder STATIC bert_tokenizer_decoder.cc)
set_property(TARGET bert_tokenizer_decoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET bert_tokenizer_decoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(bert_tokenizer_decoder PUBLIC ustring string_utils)

add_library(llama_tokenizer STATIC llama_tokenizer.cc)
set_property(TARGET llama_tokenizer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET llama_tokenizer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
set_property(TARGET llama_tokenizer
             PROPERTY INCLUDE_DIRECTORIES  ${sentencepiece_SOURCE_DIR}/src)
target_link_libraries(llama_tokenizer PUBLIC ustring string_utils sentencepiece-static)




