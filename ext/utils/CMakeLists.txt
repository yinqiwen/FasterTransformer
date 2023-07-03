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

add_library(time_utils STATIC time_utils.cc)
set_property(TARGET time_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET time_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(file_utils STATIC file_utils.cc)
set_property(TARGET file_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET file_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(string_utils STATIC string_utils.cc)
set_property(TARGET string_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET string_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(spdlogger STATIC spdlogger.cc)
set_property(TARGET spdlogger PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET spdlogger PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(spdlogger PRIVATE spdlog::spdlog)






