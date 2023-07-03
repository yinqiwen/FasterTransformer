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
#include <string>
#include <vector>

#include "3rdparty/INIReader.h"
#include "src/fastertransformer/utils/prompt_learning.h"

namespace ft_ext {

struct ModelConfig {
    std::string model_name;
    std::string model_dir;
    std::string model_type;
    std::string data_type;

    bool sparse         = false;
    int  int8_mode      = 0;
    int  attention_type = 0;

    int tensor_para_size   = 1;
    int pipeline_para_size = 1;

    size_t head_num        = 0;
    size_t size_per_head   = 0;
    size_t vocab_size      = 0;
    size_t decoder_layers  = 0;
    size_t hidden_units    = 0;
    size_t inter_size      = 0;
    size_t max_seq_len     = 0;
    size_t max_pos_seq_len = 0;

    int start_id = 50256;
    int end_id   = 50256;

    int                                   prompt_learning_start_id  = 0;
    fastertransformer::PromptLearningType prompt_learning_type      = fastertransformer::PromptLearningType::no_prompt;
    int                                   prompt_learning_num_tasks = 0;

    float shared_contexts_ratio = 1.0;

    int enable_custom_all_reduce = 0;

    int Load(const INIReader& reader);

    std::string ToString() const
    {
        std::string s;
        s.append("model_name=" + model_name).append(",");
        s.append("model_dir=" + model_dir).append(",");
        s.append("head_num=" + std::to_string(head_num)).append(",");
        return s;
    }

    // gptVariantParams gpt_variants;
};

struct ModelServiceConfig {
    ModelConfig model;
    std::string tokenizer                              = "bert";
    size_t      worker_instance_num                    = 4;
    size_t      task_queue_fetch_size                  = 128;
    size_t      max_batch_size                         = 16;
    size_t      task_queue_wait_usecs                  = 100000;   // 100ms
    size_t      fetch_inference_instance_timeout_usecs = 2000000;  // 2000ms
    int         Load(const INIReader& reader);
};

}  // namespace ft_ext
