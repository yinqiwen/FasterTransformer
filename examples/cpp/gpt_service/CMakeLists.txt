# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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


add_executable(gpt_service_example gpt_service_example.cc)
target_link_libraries(gpt_service_example PUBLIC -lcublas -lcublasLt -lcudart -lnccl -lmpi -lmpi_cxx -lnvToolsExt -Wl,--whole-archive gpt_service gpt_service_instance -Wl,--no-whole-archive ft-ext lora_model spdlog::spdlog)

add_executable(test_lora_gemm test_lora_gemm.cc)
target_link_libraries(test_lora_gemm PUBLIC -lcublas -lcublasLt -lcudart cublasMMWrapper memory_utils lora_model)
