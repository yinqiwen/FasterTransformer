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
#include "gpt_service.h"

#include "src/fastertransformer/utils/cuda_utils.h"

#include "gpt_model.h"
#include "gpt_service_instance.h"

namespace ft_ext {
template<typename T>
int GptService<T>::OnInit()
{
    GptModel<T>* gpt_model = dynamic_cast<GptModel<T>*>(model_.get());

    std::string model_name    = config_.model.model_name;
    std::string model_variant = config_reader_.Get(model_name, "model_variant", "gpt");

    if (model_variant == "opt-pre") {
        gpt_model->gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_model->gpt_variant_params.layernorm_type             = ft::LayerNormType::pre_layernorm;
        gpt_model->gpt_variant_params.activation_type            = ft::ActivationType::Relu;
        gpt_model->gpt_variant_params.has_post_decoder_layernorm = false;
    }
    else if (model_variant == "opt-post") {
        gpt_model->gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_model->gpt_variant_params.layernorm_type             = ft::LayerNormType::post_layernorm;
        gpt_model->gpt_variant_params.activation_type            = ft::ActivationType::Relu;
        gpt_model->gpt_variant_params.has_post_decoder_layernorm = false;
    }
    else if (model_variant == "bloom-pre") {
        gpt_model->gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_model->gpt_variant_params.layernorm_type             = ft::LayerNormType::pre_layernorm;
        gpt_model->gpt_variant_params.activation_type            = ft::ActivationType::Gelu;
        gpt_model->gpt_variant_params.has_positional_encoding    = false;
        gpt_model->gpt_variant_params.has_pre_decoder_layernorm  = true;
        gpt_model->gpt_variant_params.has_post_decoder_layernorm = true;
        gpt_model->gpt_variant_params.use_attention_linear_bias  = true;
    }
    else if (model_variant == "bloom-post") {
        gpt_model->gpt_variant_params.layernorm_eps              = 1e-5f;
        gpt_model->gpt_variant_params.layernorm_type             = ft::LayerNormType::post_layernorm;
        gpt_model->gpt_variant_params.activation_type            = ft::ActivationType::Gelu;
        gpt_model->gpt_variant_params.has_positional_encoding    = false;
        gpt_model->gpt_variant_params.has_pre_decoder_layernorm  = true;
        gpt_model->gpt_variant_params.has_post_decoder_layernorm = true;
        gpt_model->gpt_variant_params.use_attention_linear_bias  = true;
    }
    else {
        /* Meta Opt Examples
      layernorm_eps=1e-5
      layernorm_type=pre_layernorm
      activation_type=Relu
      has_post_decoder_layernorm=0
      */
        gpt_model->gpt_variant_params.layernorm_eps = config_reader_.GetFloat("gpt", "layernorm_eps", 1e-6f);
        gpt_model->gpt_variant_params.layernorm_type =
            ft::getLayerNormType(config_reader_.Get("gpt", "layernorm_type", "pre_layernorm"));
        gpt_model->gpt_variant_params.activation_type =
            ft::getActivationType(config_reader_.Get("gpt", "activation_type", "Gelu"));

        gpt_model->gpt_variant_params.has_positional_encoding =
            config_reader_.GetBoolean("gpt", "has_positional_encoding", true);
        gpt_model->gpt_variant_params.has_pre_decoder_layernorm =
            config_reader_.GetBoolean("gpt", "has_pre_decoder_layernorm", false);
        gpt_model->gpt_variant_params.has_post_decoder_layernorm =
            config_reader_.GetBoolean("gpt", "has_post_decoder_layernorm", true);
        gpt_model->gpt_variant_params.use_attention_linear_bias =
            config_reader_.GetBoolean("gpt", "use_attention_linear_bias", false);
    }

    gpt_model->gpt_variant_params.has_adapters = config_reader_.GetBoolean(model_name, "has_adapters", false);

    // Prompt Learning Configurations
    gpt_model->prompt_learning_start_id =
        config_reader_.GetInteger(model_name, "prompt_learning_start_id", config_.model.end_id + 1);
    gpt_model->prompt_learning_type =
        static_cast<ft::PromptLearningType>(config_reader_.GetInteger(model_name, "prompt_learning_type", 0));

    // NOTE: get prompt from configuration files
    int num_tasks = config_reader_.GetInteger(model_name, "num_tasks", 0);
    for (int task_name_id = 0; task_name_id < num_tasks; task_name_id++) {
        std::string config_task_name = model_name + "_task_" + std::to_string(task_name_id);
        std::string task_name        = config_reader_.Get(config_task_name, "task_name");
        const int   prompt_length    = config_reader_.GetInteger(config_task_name, "prompt_length", 0);
        gpt_model->prompt_learning_table_pair.insert({task_name, {task_name_id, prompt_length}});
    }

    config_.model.max_seq_len = gpt_model->gpt_variant_params.has_positional_encoding ?
                                    config_reader_.GetInteger("ft_instance_hyperparameter", "max_seq_len") :
                                    FT_SEQ_LEN_MAX;

    ft::AttentionType attention_type = ft::getAttentionType<T>(config_.model.size_per_head,
                                                               ft::getSMVersion(),
                                                               true,
                                                               0,  // gpt supports any-seq-length fmha
                                                               config_.model.int8_mode != 2,  // is_fuse
                                                               false,  // with_relative_position_bias
                                                               true);  // causal_mask
    config_.model.attention_type     = static_cast<int>(attention_type);
    FT_INFO("GPT model has_pre_decoder_layernorm:{},has_positional_encoding:{},has_post_decoder_layernorm:{}",
            gpt_model->gpt_variant_params.has_pre_decoder_layernorm,
            gpt_model->gpt_variant_params.has_positional_encoding,
            gpt_model->gpt_variant_params.has_post_decoder_layernorm);
    // load weights

    int device_id = 0;
    for (int tensor_rank = 0; tensor_rank < config_.model.tensor_para_size; tensor_rank++) {
        for (int pipeline_rank = 0; pipeline_rank < config_.model.pipeline_para_size; pipeline_rank++) {
            ft::check_cuda_error(cudaSetDevice(device_id));
            auto weight = std::make_unique<ft::ParallelGptWeight<T>>(config_.model.hidden_units,
                                                                     config_.model.inter_size,
                                                                     config_.model.vocab_size,
                                                                     config_.model.decoder_layers,
                                                                     config_.model.max_seq_len,
                                                                     config_.model.tensor_para_size,
                                                                     tensor_rank,
                                                                     config_.model.pipeline_para_size,
                                                                     pipeline_rank,
                                                                     config_.model.int8_mode,
                                                                     gpt_model->prompt_learning_type,
                                                                     gpt_model->prompt_learning_table_pair,
                                                                     gpt_model->gpt_variant_params);

            weight->loadModel(config_.model.model_dir);
            weight->loadLora(config_.model.model_dir, "test_lora");

            gpt_model->weights.emplace_back(std::move(weight));
            size_t free_bytes, total_bytes;
            ft::check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
            float       free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
            float       total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
            float       used  = total - free;
            std::string time  = "after load model weights";
            FT_INFO("Device[{}] {}: tensor_rank:{}, pipeline_rank: {}, free: {} GB, total: {} GB, used: {} GB",
                    device_id,
                    time,
                    tensor_rank,
                    pipeline_rank,
                    free,
                    total,
                    used);
            device_id++;
        }
    }
    return 0;
}

template<typename T>
ModelPtr GptService<T>::NewModel()
{
    return std::make_unique<GptModel<T>>();
}
template<typename T>
ModelServiceInstancePtr GptService<T>::NewModelInstance(int                                     device_id,
                                                        const Model*                            model,
                                                        const ModelServiceConfig&               config,
                                                        ft::NcclParam                           tensor_para,
                                                        ft::NcclParam                           pipeline_para,
                                                        std::shared_ptr<ft::AbstractCustomComm> custom_comm)
{
    return std::make_shared<GptServiceInstance<T>>(device_id, model, config, tensor_para, pipeline_para, custom_comm);
}

template class GptService<float>;
template class GptService<half>;
#ifdef ENABLE_BF16
template class GptService<__nv_bfloat16>;
#endif

FT_EXT_MODEL_REG(gpt, fp32, GptService<float>)
FT_EXT_MODEL_REG(gpt, fp16, GptService<half>)
#ifdef ENABLE_BF16
FT_EXT_MODEL_REG(gpt, bf16, GptService<__nv_bfloat16>)
#endif
}  // namespace ft_ext