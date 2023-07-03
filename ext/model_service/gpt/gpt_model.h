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
#include <vector>

#include "ext/model_service/model.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"

namespace ft_ext {
template<typename T>
struct GptModel: public Model {
    using ParallelGptWeightPtr = std::unique_ptr<fastertransformer::ParallelGptWeight<T>>;
    std::vector<ParallelGptWeightPtr> weights;
    // model variants parameters
    fastertransformer::gptVariantParams        gpt_variant_params         = {};
    int                                        prompt_learning_start_id   = 0;
    ft::PromptLearningType                     prompt_learning_type       = ft::PromptLearningType::no_prompt;
    std::map<std::string, std::pair<int, int>> prompt_learning_table_pair = {};
};
}  // namespace ft_ext