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
#include <string>

#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace ft_ext {
namespace ft = fastertransformer;
template<typename T>
class LoraWeight {
public:
    LoraWeight(size_t m, size_t n, size_t k, float alpha, bool transpose);
    void load(const std::string& lora_a_file, const std::string& lora_b_file, ft::FtCudaDataType model_file_type);
    void apply(ft::cublasMMWrapper* wrapper, T* weight);
    void unapply(ft::cublasMMWrapper* wrapper, T* weight);

    ~LoraWeight();

private:
    size_t m_ = 0;
    size_t n_ = 0;
    size_t k_ = 0;

    T*    lora_A_ = nullptr;
    T*    lora_B_ = nullptr;
    float alpha_  = 0.0;

    bool transpose_ = false;
};
}  // namespace ft_ext