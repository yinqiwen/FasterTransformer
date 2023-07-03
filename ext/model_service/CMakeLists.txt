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

add_subdirectory(gpt)

add_library(model_service_instance STATIC model_service_instance.cc)
set_property(TARGET model_service_instance PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET model_service_instance PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(model_service_instance PUBLIC spdlogger time_utils)

add_library(model_service STATIC model_service.cc)
set_property(TARGET model_service PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET model_service PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(model_service PUBLIC spdlogger file_utils time_utils)





