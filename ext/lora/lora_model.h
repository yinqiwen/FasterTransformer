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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lora_weight.h"

#include "src/fastertransformer/utils/cuda_utils.h"

namespace ft_ext {

using LayerName = std::pair<int, std::string>;
struct LayerNameHash {
    std::size_t operator()(const LayerName& p) const
    {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2;
    }
};

struct WeightOptions {
    std::vector<int> shape;
    bool             transpose = false;
};

using GetWeightOptionsByName = std::function<WeightOptions(const std::string&, int)>;
using GetWeightByName        = std::function<void*(const std::string&, int)>;

template<typename T>
class LoraModel {
public:
    LoraModel(const GetWeightOptionsByName& get_weight_opt, const GetWeightByName& get_weight);
    int load(const std::string& dir, ft::FtCudaDataType model_file_type);
    int apply(ft::cublasMMWrapper* wrapper);
    int unapply(ft::cublasMMWrapper* wrapper);

private:
    std::unordered_map<LayerName, std::unique_ptr<LoraWeight<T>>, LayerNameHash> weights_;
    GetWeightOptionsByName                                                       get_weight_opt_;
    GetWeightByName                                                              get_weight_;
    int                                                                          lora_r_     = 0;
    int                                                                          lora_alpha_ = 0;
};
}  // namespace ft_ext