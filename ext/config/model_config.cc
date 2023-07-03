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
#include "model_config.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/logger.h"

#define FT_SEQ_LEN_MAX (16384)

namespace ft_ext {

int ModelConfig::Load(const INIReader& reader)
{
    model_type = reader.Get("ft_instance_hyperparameter", "model_type");
    model_name = reader.Get("ft_instance_hyperparameter", "model_name");
    model_dir  = std::string(reader.Get("ft_instance_hyperparameter", "model_dir"));
    data_type  = reader.Get("ft_instance_hyperparameter", "data_type");
    sparse     = static_cast<bool>(reader.GetInteger("ft_instance_hyperparameter", "sparse"));
    int8_mode  = reader.GetInteger("ft_instance_hyperparameter", "int8_mode");

    tensor_para_size   = reader.GetInteger("ft_instance_hyperparameter", "tensor_para_size");
    pipeline_para_size = reader.GetInteger("ft_instance_hyperparameter", "pipeline_para_size");

    max_seq_len = (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len");

    shared_contexts_ratio = reader.GetFloat("ft_instance_hyperparameter", "shared_contexts_ratio", 1.0f);

    enable_custom_all_reduce = reader.GetInteger("ft_instance_hyperparameter", "enable_custom_all_reduce");

    std::string model_config_path = model_dir + "/config.ini";
    INIReader   model_conf_reader = INIReader(model_config_path);
    if (model_conf_reader.ParseError() < 0) {
        FT_ERROR("Can't load {}", model_config_path);
        return -1;
    }

    // std::string model_variant = std::string(reader.Get(model_name, "model_variant", "gpt"));

    // max_seq_len = has_positional_encoding(model_variant) ?
    //                   (size_t)reader.GetInteger("ft_instance_hyperparameter", "max_seq_len") :
    //                   FT_SEQ_LEN_MAX;
    head_num        = model_conf_reader.GetInteger(model_type, "head_num");
    size_per_head   = model_conf_reader.GetInteger(model_type, "size_per_head");
    vocab_size      = model_conf_reader.GetInteger(model_type, "vocab_size");
    decoder_layers  = model_conf_reader.GetInteger(model_type, "num_layer");
    hidden_units    = head_num * size_per_head;
    inter_size      = model_conf_reader.GetInteger(model_type, "inter_size", 4 * hidden_units);
    start_id        = model_conf_reader.GetInteger(model_type, "start_id", 50256);
    end_id          = model_conf_reader.GetInteger(model_type, "end_id", 50256);
    max_pos_seq_len = model_conf_reader.GetInteger(model_type, "max_pos_seq_len", 1024);

    // Prompt Learning Configurations
    prompt_learning_start_id = reader.GetInteger(model_name, "prompt_learning_start_id", end_id + 1);
    prompt_learning_type =
        static_cast<fastertransformer::PromptLearningType>(reader.GetInteger(model_name, "prompt_learning_type", 0));
    prompt_learning_num_tasks = reader.GetInteger(model_name, "num_tasks", 0);

    return 0;
}
int ModelServiceConfig::Load(const INIReader& reader)
{
    worker_instance_num   = reader.GetInteger("service", "worker_instance_num", 4);
    task_queue_wait_usecs = reader.GetInteger("service", "task_queue_wait_usecs", 10000);
    fetch_inference_instance_timeout_usecs =
        reader.GetInteger("service", "fetch_inference_instance_timeout_usecs", 2000000);
    task_queue_fetch_size = reader.GetInteger("service", "task_queue_fetch_size", 128);
    max_batch_size        = reader.GetInteger("service", "max_batch_size", 16);
    if (task_queue_fetch_size < max_batch_size) {
        max_batch_size = max_batch_size;
    }
    tokenizer = reader.Get("service", "tokenizer");
    return model.Load(reader);
}

}  // namespace ft_ext