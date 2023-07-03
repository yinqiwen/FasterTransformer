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
#include <stdio.h>

#include <deque>
#include <set>
#include <string>
#include <vector>

namespace ft_ext {
int  file_read_all(const std::string& path, std::string& content);
bool is_file_exist(const std::string& path);
bool is_dir_exist(const std::string& path);
int  list_subfiles(const std::string& path, std::vector<std::string>& fs);

bool make_file(const std::string& para_path);
bool make_dir(const std::string& para_path);

int file_read_content(const std::string& path, void* buf, uint32_t size);
int file_write_content(const std::string& path, const std::string& content);
int file_write_content(const std::string& path, const void* content, size_t size);

int64_t file_size(const std::string& path);
}  // namespace ft_ext