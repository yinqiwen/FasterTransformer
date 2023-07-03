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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace ft_ext {
struct llama_vocab;
class LLamaTokenizer {
public:
    LLamaTokenizer();
    int              Load(const std::string& model, const std::string& added_tokens);
    std::vector<int> Encode(const std::string& str, bool add_bos = true);
    std::string      Decode(const std::vector<int>& ids);

private:
    std::shared_ptr<sentencepiece::SentencePieceProcessor> processor_;
};
}  // namespace ft_ext
