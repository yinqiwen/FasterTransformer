/*
 * Copyright (c) 2022 TENCENT CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <stdint.h>
#include <string>
#include <sys/time.h>
#include <time.h>
namespace ft_ext {
int64_t gettimeofday_us();
int64_t gettimeofday_ms();
int64_t gettimeofday_s();
time_t  fast_mktime(struct tm* tm);
int64_t unixsec_from_timefield(const std::string& time_field);
}  // namespace ft_ext