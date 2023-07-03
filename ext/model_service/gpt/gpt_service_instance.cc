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
#include "gpt_service_instance.h"

#include <exception>

#include "src/fastertransformer/utils/memory_utils.h"

namespace ft_ext {
static void streamingCallback(std::unordered_map<std::string, ft::Tensor>* tensors, void* cb_data)
{
    std::pair<ModelServiceInstance*, void*>* model_task =
        reinterpret_cast<std::pair<ModelServiceInstance*, void*>*>(cb_data);
    ModelServiceInstance*       instance = model_task->first;
    ModelInstanceInferenceTask* task     = reinterpret_cast<ModelInstanceInferenceTask*>(model_task->second);
    instance->StreamingGenerateCallback(*task);
}

template<typename T>
GptServiceInstance<T>::GptServiceInstance(int                                     device_id,
                                          const Model*                            model,
                                          const ModelServiceConfig&               config,
                                          ft::NcclParam                           tensor_para,
                                          ft::NcclParam                           pipeline_para,
                                          std::shared_ptr<ft::AbstractCustomComm> custom_comm):
    ModelServiceInstance(device_id, model, config, tensor_para, pipeline_para, custom_comm)
{
    gpt_model_ = dynamic_cast<const GptModel<T>*>(model);
    auto* gpt  = new ft::ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                       0,  // max_seq_len, FT will adjust the buffer automatically.
                                       0,  // max_input_len, FT will adjust the buffer automatically.
                                       0,
                                       config_.model.head_num,
                                       config_.model.size_per_head,
                                       config_.model.inter_size,
                                       config_.model.decoder_layers,
                                       0,   // expert_num
                                       0,   // moe_k
                                       {},  // moe_layer_index
                                       config_.model.vocab_size,
                                       config_.model.start_id,
                                       config_.model.end_id,
                                       gpt_model_->prompt_learning_start_id,  // p/prompt tuning virtual token start id
                                       gpt_model_->prompt_learning_type,
                                       gpt_model_->gpt_variant_params,
                                       0.0f,  // beam_search_diversity_rate_,
                                       1,     // top_k_,
                                       0.0f,  // top_p_,
                                       0,     // random seed, note that all gpus should use same seed
                                       1.0f,  // temperature_,
                                       0.0f,  // len_penalty_,
                                       1.0f,  // repetition_penalty_,
                                       tensor_para,
                                       pipeline_para,
                                       stream_,
                                       cublas_wrapper_.get(),
                                       allocator_.get(),
                                       false,
                                       &device_prop_,
                                       (ft::AttentionType)config_.model.attention_type,
                                       false,
                                       config_.model.int8_mode,
                                       custom_comm_,
                                       config_.model.enable_custom_all_reduce);
    gpt_.reset(gpt);
}
template<typename T>
int GptServiceInstance<T>::DoInference(std::unordered_map<std::string, ft::Tensor>& input_tensors,
                                       std::unordered_map<std::string, ft::Tensor>& output_tensors,
                                       bool                                         stream)
{

    if (stream) {
        gpt_->registerCallback(streamingCallback, &streaming_cbdata_);
    }
    try {
        // if (stream_cb_ != nullptr) {
        //     gpt_->registerCallback(triton_stream_callback<T>, this);
        // }
        gpt_->forward(&output_tensors, &input_tensors, gpt_model_->weights[device_id_].get());
        // gpt_->unRegisterCallback();
    }
    catch (...) {
        std::exception_ptr ex = std::current_exception();
        try {
            std::rethrow_exception(ex);
        }
        catch (const std::exception& e) {
            FT_ERROR("Caught exception: {}", e.what());
        }
        // output_tensors.insert({"error_message", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BYTES, {1}, &h_exception_}});
        gpt_->unRegisterCallback();
        return -1;
    }
    gpt_->unRegisterCallback();
    return 0;
}
template<typename T>
int GptServiceInstance<T>::PrepareInferenceTensors(ModelInstanceInferenceTask&                  task,
                                                   std::unordered_map<std::string, ft::Tensor>& input_tensors,
                                                   std::unordered_map<std::string, ft::Tensor>& output_tensors)
{
    if (task.stream) {
        streaming_cbdata_.first  = this;
        streaming_cbdata_.second = &task;
    }
    const size_t request_batch_size = task.batch.size();
    size_t       beam_width         = task.batch[0].request.beam_width;

    if (beam_width != 1 && beam_width != 2 && beam_width != 3 && beam_width != 4 && beam_width != 8 && beam_width != 16
        && beam_width != 32) {
        FT_WARN("beam_width = {} is invalid. Set it to 1 to use sampling by default.", beam_width);
        beam_width = 1;
    }
    size_t total_length = task.max_request_output_len + task.max_input_len;
    AllocateBuffer(request_batch_size, beam_width, total_length, task.max_request_output_len);
    d_input_ids_ = reinterpret_cast<int*>(
        allocator_->reMalloc(d_input_ids_, sizeof(int) * request_batch_size * task.max_input_len, false));
    d_input_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(d_input_lengths_, sizeof(int) * request_batch_size, false));

    ft::cudaAutoCpy(d_input_ids_, task.start_ids.data(), request_batch_size * task.max_input_len, stream_);
    ft::cudaAutoCpy(d_input_lengths_, task.start_lengths.data(), request_batch_size, stream_);

    input_tensors.emplace("input_ids",
                          ft::Tensor{ft::MEMORY_GPU,
                                     ft::TYPE_INT32,
                                     std::vector<size_t>{request_batch_size, (size_t)task.max_input_len},
                                     d_input_ids_});
    input_tensors.emplace(
        "input_lengths",
        ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, d_input_lengths_});
    input_tensors.emplace(
        "output_seq_len",
        ft::Tensor{
            ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, task.output_seq_len.data()});
    input_tensors.emplace(
        "temperature",
        ft::Tensor{
            ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{task.temperature.size()}, task.temperature.data()});
    input_tensors.emplace(
        "len_penalty",
        ft::Tensor{
            ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{task.len_penalty.size()}, task.len_penalty.data()});
    if (!task.min_length.empty()) {
        input_tensors.emplace(
            "min_length",
            ft::Tensor{
                ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{task.min_length.size()}, task.min_length.data()});
    }
    if (!task.start_id.empty()) {
        input_tensors.emplace(
            "start_id",
            ft::Tensor{
                ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{task.start_id.size()}, task.start_id.data()});
    }
    if (!task.end_id.empty()) {
        input_tensors.emplace(
            "end_id",
            ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{task.end_id.size()}, task.end_id.data()});
    }

    if (!task.repetition_penalty.empty()) {
        input_tensors.emplace("repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU,
                                         ft::TYPE_UINT32,
                                         std::vector<size_t>{task.repetition_penalty.size()},
                                         task.repetition_penalty.data()});
    }
    if (!task.presence_penalty.empty()) {
        input_tensors.emplace("presence_penalty",
                              ft::Tensor{ft::MEMORY_CPU,
                                         ft::TYPE_UINT32,
                                         std::vector<size_t>{task.presence_penalty.size()},
                                         task.presence_penalty.data()});
    }

    if (!task.runtime_top_k.empty()) {
        input_tensors.emplace("runtime_top_k",
                              ft::Tensor{ft::MEMORY_CPU,
                                         ft::TYPE_UINT32,
                                         std::vector<size_t>{task.runtime_top_k.size()},
                                         task.runtime_top_k.data()});
    }
    if (!task.runtime_top_p.empty()) {
        input_tensors.emplace("runtime_top_p",
                              ft::Tensor{ft::MEMORY_CPU,
                                         ft::TYPE_FP32,
                                         std::vector<size_t>{task.runtime_top_p.size()},
                                         task.runtime_top_p.data()});
    }
    if (!task.beam_search_diversity_rate.empty()) {
        input_tensors.emplace("beam_search_diversity_rate",
                              ft::Tensor{ft::MEMORY_CPU,
                                         ft::TYPE_FP32,
                                         std::vector<size_t>{task.beam_search_diversity_rate.size()},
                                         task.beam_search_diversity_rate.data()});
    }

    output_tensors.emplace("output_ids",
                           ft::Tensor{ft::MEMORY_GPU,
                                      ft::TYPE_UINT32,
                                      std::vector<size_t>{request_batch_size, beam_width, total_length},
                                      d_output_ids_});
    output_tensors.emplace(
        "sequence_length",
        ft::Tensor{
            ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size, beam_width}, d_sequence_lengths_});
    output_tensors.emplace("response_input_lengths",
                           ft::Tensor{ft::MEMORY_GPU,
                                      ft::TYPE_INT32,
                                      std::vector<size_t>{request_batch_size, beam_width},
                                      d_response_input_lengths_});
    output_tensors.emplace(
        "is_finished",
        ft::Tensor{ft::MEMORY_GPU, ft::TYPE_BOOL, std::vector<size_t>{request_batch_size, beam_width}, d_is_finished_});

    return 0;
}
template<typename T>
void GptServiceInstance<T>::GetOutput(size_t                         request_batch_size,
                                      size_t                         beam_width,
                                      size_t                         max_input_len,
                                      size_t                         max_request_output_len,
                                      const std::vector<int>&        ignore_prefix_lengths,
                                      std::vector<std::vector<int>>& output_ids)
{
    const int        output_unit_len = max_input_len + max_request_output_len;
    std::vector<int> output_buf(output_unit_len * request_batch_size * beam_width);

    ft::cudaD2Hcpy(&output_buf[0], d_output_ids_, output_buf.size());

    size_t           output_length_size = request_batch_size * beam_width;
    std::vector<int> len_buf(output_length_size);
    ft::cudaD2Hcpy(&len_buf[0], d_sequence_lengths_, output_length_size);

    output_ids.clear();
    for (size_t i = 0; i < len_buf.size(); i++) {
        const int*       data                 = output_buf.data() + i * output_unit_len;
        int              batch_idx            = i / beam_width;
        int              ignore_prefix_length = ignore_prefix_lengths[batch_idx];
        std::vector<int> ids(data + ignore_prefix_length, data + len_buf[i]);
        output_ids.emplace_back(std::move(ids));
    }
}

template<typename T>
void GptServiceInstance<T>::AllocateBuffer(const size_t request_batch_size,
                                           const size_t beam_width,
                                           const size_t total_output_len,
                                           const size_t request_output_len)
{

    d_output_ids_ = (int*)(allocator_->reMalloc(
        d_output_ids_, sizeof(int) * request_batch_size * beam_width * total_output_len, false));
    d_sequence_lengths_ =
        (int*)(allocator_->reMalloc(d_sequence_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_response_input_lengths_ =
        (int*)(allocator_->reMalloc(d_response_input_lengths_, sizeof(int) * request_batch_size * beam_width, false));
    d_output_log_probs_ = (float*)(allocator_->reMalloc(
        d_output_log_probs_, sizeof(float) * request_batch_size * beam_width * request_output_len, false));
    d_output_ctx_emb_   = (float*)(allocator_->reMalloc(
        d_output_ctx_emb_, sizeof(float) * request_batch_size * beam_width * gpt_->getHiddenUnits(), false));
    d_cum_log_probs_ =
        (float*)(allocator_->reMalloc(d_cum_log_probs_, sizeof(float) * request_batch_size * beam_width, false));
    d_is_finished_ =
        (bool*)(allocator_->reMalloc(d_is_finished_, sizeof(bool) * request_batch_size * beam_width, false));
}

template<typename T>
void GptServiceInstance<T>::FreeBuffer()
{
    allocator_->free((void**)(&d_output_ids_));
    allocator_->free((void**)(&d_sequence_lengths_));
    allocator_->free((void**)(&d_response_input_lengths_));
    allocator_->free((void**)(&d_output_log_probs_));
    allocator_->free((void**)(&d_output_ctx_emb_));
    allocator_->free((void**)(&d_cum_log_probs_));
    allocator_->free((void**)(&d_is_finished_));

    allocator_->free((void**)(&d_input_ids_));
    allocator_->free((void**)(&d_input_lengths_));
}

template class GptServiceInstance<float>;
template class GptServiceInstance<half>;
#ifdef ENABLE_BF16
template class GptServiceInstance<__nv_bfloat16>;
#endif
}  // namespace ft_ext