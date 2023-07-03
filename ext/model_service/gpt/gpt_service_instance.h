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
#include <utility>
#include <vector>

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"

#include "ext/model_service/model_service_instance.h"

#include "gpt_model.h"

namespace ft_ext {
template<typename T>
class GptServiceInstance: public ModelServiceInstance {
public:
    GptServiceInstance(int                                     device_id,
                       const Model*                            model,
                       const ModelServiceConfig&               config,
                       ft::NcclParam                           tensor_para,
                       ft::NcclParam                           pipeline_para,
                       std::shared_ptr<ft::AbstractCustomComm> custom_comm);

private:
    int PrepareInferenceTensors(ModelInstanceInferenceTask&                  tasks,
                                std::unordered_map<std::string, ft::Tensor>& input,
                                std::unordered_map<std::string, ft::Tensor>& output) override;
    int DoInference(std::unordered_map<std::string, ft::Tensor>& input,
                    std::unordered_map<std::string, ft::Tensor>& output,
                    bool                                         stream) override;

    void GetOutput(size_t                         request_batch_size,
                   size_t                         beam_width,
                   size_t                         max_input_len,
                   size_t                         max_request_output_len,
                   const std::vector<int>&        ignore_prefix_lengths,
                   std::vector<std::vector<int>>& output_ids) override;

    void AllocateBuffer(const size_t request_batch_size,
                        const size_t beam_width,
                        const size_t total_output_len,
                        const size_t request_output_len);
    void FreeBuffer();

    const GptModel<T>*                  gpt_model_ = nullptr;
    std::unique_ptr<ft::ParallelGpt<T>> gpt_;

    int*   d_input_ids_                = nullptr;
    int*   d_input_lengths_            = nullptr;
    int*   d_request_prompt_lengths_   = nullptr;
    int*   d_input_bad_words_          = nullptr;
    int*   d_input_stop_words_         = nullptr;
    T*     d_request_prompt_embedding_ = nullptr;
    float* d_top_p_decay_              = nullptr;
    float* d_top_p_min_                = nullptr;
    int*   d_top_p_reset_ids_          = nullptr;

    int*   d_output_ids_             = nullptr;
    int*   d_sequence_lengths_       = nullptr;
    int*   d_response_input_lengths_ = nullptr;
    float* d_output_log_probs_       = nullptr;
    float* d_cum_log_probs_          = nullptr;
    float* d_output_ctx_emb_         = nullptr;
    bool*  d_is_finished_            = nullptr;

    // uint32_t*             h_total_output_lengths_ = nullptr;
    std::exception_ptr h_exception_ = nullptr;

    std::pair<ModelServiceInstance*, void*> streaming_cbdata_;
};
}  // namespace ft_ext