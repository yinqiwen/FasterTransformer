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


add_library(lora_model STATIC lora_model.cc lora_weight.cc)
set_property(TARGET lora_model PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET lora_model PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(lora_model PUBLIC spdlogger file_utils string_utils cuda_utils memory_utils)






