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
#include "lora_model.h"
#include <string_view>

#include "3rdparty/INIReader.h"
#include "ext/utils/file_utils.h"
#include "ext/utils/spdlogger.h"
#include "ext/utils/string_utils.h"

namespace ft_ext {
static constexpr std::string_view kLoraBWeightSuffix = ".lora_B.weight.bin";
static constexpr std::string_view kLoraAWeightSuffix = ".lora_A.weight.bin";
static constexpr std::string_view kLoraConfigFile    = "adapter_config.ini";
template<typename T>
LoraModel<T>::LoraModel(const GetWeightOptionsByName& get_weight_opt, const GetWeightByName& get_weight):
    get_weight_opt_(get_weight_opt), get_weight_(get_weight)
{
}

template<typename T>
int LoraModel<T>::load(const std::string& dir, ft::FtCudaDataType model_file_type)
{
    std::string config_file = dir + "/" + std::string(kLoraConfigFile);
    INIReader   reader      = INIReader(config_file);
    if (reader.ParseError() < 0) {
        FT_ERROR("Can't load {}", config_file);
        return -1;
    }
    lora_r_     = reader.GetInteger("lora", "r");
    lora_alpha_ = reader.GetInteger("lora", "lora_alpha");

    std::vector<std::string> all_files;
    list_subfiles(dir, all_files);

    std::vector<std::string> weight_names;
    for (const std::string& fname : all_files) {
        if (has_suffix(fname, std::string(kLoraBWeightSuffix))) {
            std::string weight_name = fname.substr(0, fname.length() - kLoraBWeightSuffix.length());
            weight_names.emplace_back(weight_name);
        }
    }

    for (size_t i = 0; i < weight_names.size(); i++) {
        const std::string& weight_name = weight_names[i];
        std::string        lora_a_file = dir + "/" + weight_name + std::string(kLoraAWeightSuffix);
        std::string        lora_b_file = dir + "/" + weight_name + std::string(kLoraBWeightSuffix);
        if (!is_file_exist(lora_a_file)) {
            FT_ERROR("{} is not exist!", lora_a_file);
            return -1;
        }
        if (!is_file_exist(lora_b_file)) {
            FT_ERROR("{} is not exist!", lora_b_file);
            return -1;
        }
        int layer = -1;
        if (has_prefix(weight_name, "h.")) {
            std::string tmp = weight_name.substr(2);
            layer           = std::stoi(tmp);
        }

        WeightOptions opt = get_weight_opt_(weight_name, layer);
        if (opt.shape.size() != 2) {
            FT_ERROR("Invalid shape size:{} for weight name:{}", opt.shape.size(), weight_name);
            return -1;
        }
        size_t lora_b_size = file_size(lora_b_file);
        size_t lora_a_size = file_size(lora_a_file);
        if (lora_b_size != opt.shape[0] * lora_r_ * ft::cuda_datatype_size(model_file_type)) {
            FT_ERROR("Invalid lora_B file size:{} while expected:{}",
                     lora_b_size,
                     opt.shape[0] * lora_r_ * ft::cuda_datatype_size(model_file_type));
            return -1;
        }
        if (lora_a_size != opt.shape[1] * lora_r_ * ft::cuda_datatype_size(model_file_type)) {
            FT_ERROR("Invalid lora_A file size:{} while expected:{}",
                     lora_a_size,
                     opt.shape[1] * lora_r_ * ft::cuda_datatype_size(model_file_type));
            return -1;
        }
        LayerName key;
        key.first  = layer;
        key.second = weight_name;

        std::unique_ptr<LoraWeight<T>> lora_weight = std::make_unique<LoraWeight<T>>(
            opt.shape[0], lora_r_, opt.shape[1], lora_alpha_ * 1.0 / lora_r_, opt.transpose);
        lora_weight->load(lora_a_file, lora_b_file, model_file_type);
        weights_.emplace(key, std::move(lora_weight));
    }
    return 0;
}
template<typename T>
int LoraModel<T>::apply(ft::cublasMMWrapper* wrapper)
{
    for (auto& [key, w] : weights_) {
        const auto& [layer, name] = key;
        void* orig_weight         = get_weight_(name, layer);
        if (nullptr == orig_weight) {
            FT_ERROR("No orig weight found for {} to apply lora.", name);
            return -1;
        }
    }
    return 0;
}
template<typename T>
int LoraModel<T>::unapply(ft::cublasMMWrapper* wrapper)
{
    for (auto& [key, w] : weights_) {
        const auto& [layer, name] = key;
        void* orig_weight         = get_weight_(name, layer);
        if (nullptr == orig_weight) {
            FT_ERROR("No orig weight found for {} to unapply lora.", name);
            return -1;
        }
    }
    return 0;
}

template class LoraModel<float>;
template class LoraModel<half>;
#ifdef ENABLE_BF16
template class LoraModel<__nv_bfloat16>;
#endif

}  // namespace ft_ext