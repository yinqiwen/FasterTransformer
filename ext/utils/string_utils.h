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
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ft_ext {
//====copy from  https://github.com/microsoft/onnxruntime-extensions

template<typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept
{
    ss << t;
}

template<>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept
{
    ss << "[";
    for (size_t i = 0; i < t.size(); i++) {
        if (i != 0) {
            ss << ", ";
        }
        ss << t[i];
    }
    ss << "]";
}

// template<>
// inline void MakeStringInternal(std::ostringstream& ss, const OrtTensorDimensions& t) noexcept
// {
//     MakeStringInternal(ss, static_cast<const std::vector<int64_t>&>(t));
// }

template<>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<std::string>& t) noexcept
{
    ss << "[";
    for (size_t i = 0; i < t.size(); i++) {
        if (i != 0) {
            ss << ", ";
        }
        ss << t[i];
    }
    ss << "]";
}

template<typename T, typename... Args>
void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept
{
    MakeStringInternal(ss, t);
    MakeStringInternal(ss, args...);
}

template<typename... Args>
std::string MakeString(const Args&... args)
{
    std::ostringstream ss;
    MakeStringInternal(ss, args...);
    return std::string(ss.str());
}

std::vector<std::string_view>
SplitString(const std::string_view& str, const std::string_view& seps, bool remove_empty_entries = false);

bool IsCJK(char32_t c);

bool IsAccent(char32_t c);

bool IsSpace(char32_t c);

bool IsPunct(char32_t c);

bool IsControl(char32_t c);

char32_t ToLower(char32_t c);

char32_t StripAccent(char32_t c);

uint64_t Hash64(const char* data, size_t n, uint64_t seed);

inline uint64_t Hash64(const char* data, size_t n)
{
    return Hash64(data, n, 0xDECAFCAFFE);
}

uint64_t Hash64Fast(const char* data, size_t n);

bool has_prefix(const std::string& str, const std::string& prefix);
bool has_suffix(const std::string& str, const std::string& suffix);
}  // namespace ft_ext