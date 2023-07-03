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
#include "model_service_instance.h"

#include "src/fastertransformer/utils/cuda_utils.h"

#include "ext/utils/time_utils.h"

#include <cuda_runtime.h>

namespace ft_ext {
ModelServiceInstance::ModelServiceInstance(int                                     device_id,
                                           const Model*                            model,
                                           const ModelServiceConfig&               config,
                                           ft::NcclParam                           tensor_para,
                                           ft::NcclParam                           pipeline_para,
                                           std::shared_ptr<ft::AbstractCustomComm> custom_comm):
    device_id_(device_id),
    model_(model),
    config_(config),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_comm_(custom_comm)
{
    ft::check_cuda_error(cudaSetDevice(device_id));
    ft::check_cuda_error(cudaGetDeviceProperties(&device_prop_, device_id));

    allocator_ = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(device_id);
    cudaStreamCreate(&stream_);

    allocator_->setStream(stream_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
    cublasSetStream(cublas_handle_, stream_);

#ifdef SPARSITY_ENABLED
    CHECK_CUSPARSE(cusparseLtInit(&cusparselt_handle_));
    cublasAlgoMap* cublas_algo_map = new cublasAlgoMap(GEMM_CONFIG, SPGEMM_CONFIG);
#else
    cublas_algo_map_ = std::make_unique<ft::cublasAlgoMap>(GEMM_CONFIG);
#endif

#ifdef SPARSITY_ENABLED
    cublas_wrapper_ = std::make_unique<cublasMMWrapper>(cublas_handle_,
                                                        cublaslt_handle_,
                                                        cusparselt_handle_,
                                                        stream_,
                                                        cublas_algo_map_,
                                                        &cublas_wrapper_mutex,
                                                        &allocator);
#else
    cublas_wrapper_  = std::make_unique<ft::cublasMMWrapper>(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(), &cublas_wrapper_mutex_, allocator_.get());
#endif
    cublas_wrapper_->setStream(stream_);
    if (config_.model.data_type == "fp16") {
        cublas_wrapper_->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
#ifdef ENABLE_BF16
    else if (config_.model.data_type == "bf16") {
        cublas_wrapper_->setBF16GemmConfig();
    }
#endif
    else if (config_.model.data_type == "fp32") {
        cublas_wrapper_->setFP32GemmConfig();
    }
    cudaDeviceSynchronize();

    running_                   = true;
    inference_thread_          = std::make_unique<std::thread>(&ModelServiceInstance::RunInference, this);
    inference_callback_thread_ = std::make_unique<std::thread>(&ModelServiceInstance::RunInferenceCallback, this);
}

ModelServiceInstance::~ModelServiceInstance()
{
    running_ = false;
    if (inference_thread_) {
        inference_thread_->join();
    }
    if (inference_callback_thread_) {
        inference_callback_thread_->join();
    }
}

bool ModelServiceInstance::Post(const ModelInstanceInferenceTask& tasks)
{
    bool r = inference_queue_.try_enqueue(tasks);
    if (!r) {
        FT_ERROR("Failed to enqueue generate task with queue appromix size:{}", inference_queue_.size_approx());
    }
    return r;
}

bool ModelServiceInstance::Post(ModelInstanceInferenceTask&& tasks)
{
    bool r = inference_queue_.try_enqueue(std::move(tasks));
    if (!r) {
        FT_ERROR("Failed to enqueue generate task with queue appromix size:{}", inference_queue_.size_approx());
    }
    return r;
}

void ModelServiceInstance::FillGenerateResponse(const ModelInstanceInferenceTask&    tasks,
                                                const std::vector<std::vector<int>>& output_ids,
                                                std::vector<GenerateResponse>&       batch_response)
{
    size_t task_idx   = 0;
    size_t beam_idx   = 0;
    size_t beam_width = tasks.batch[0].request.beam_width;
    for (const auto& ids : output_ids) {
        std::string decode_txt = model_->tokenizer_decoder->Decode(ids, false, true);
        batch_response[task_idx].choices.emplace_back(decode_txt);
        batch_response[task_idx].stat = tasks.stats[task_idx];
        beam_idx++;
        if (beam_idx >= beam_width) {
            beam_idx = 0;
            task_idx++;
        }
    }
}

void ModelServiceInstance::StreamingGenerateCallback(ModelInstanceInferenceTask& task)
{
    TaskCallback(task, false);
    for (int& len : task.stream_gen_lengths) {
        len++;
    }
}

void ModelServiceInstance::TaskCallback(const ModelInstanceInferenceTask& task, bool complete)
{
    ft::sync_check_cuda_error();
    cudaStreamSynchronize(stream_);

    std::vector<std::vector<int>> output_ids;
    std::vector<GenerateResponse> batch_response(task.batch.size());
    GetOutput(task.batch.size(),
              task.batch[0].request.beam_width,
              task.max_input_len,
              task.max_request_output_len,
              task.stream ? task.stream_gen_lengths : task.start_lengths,
              output_ids);
    FillGenerateResponse(task, output_ids, batch_response);
    for (size_t i = 0; i < task.batch.size(); i++) {
        batch_response[i].complete = complete;
        if (complete) {
            batch_response[i].stat.complete_us = gettimeofday_us();
        }
        PostCallback(task.batch[i].callback, std::move(batch_response[i]));
    }
}

bool ModelServiceInstance::PostCallback(const GenerateCallback& cb, GenerateResponse&& response)
{
    GenerateCallbackTask task;
    task.callback = cb;
    task.response = std::move(response);
    bool r        = inference_callback_queue_.try_enqueue(std::move(task));
    if (!r) {
        FT_ERROR("Failed to enqueue callback task with queue appromix size:{}",
                 inference_callback_queue_.size_approx());
    }
    return r;
}

void ModelServiceInstance::RunInference()
{
    ft::check_cuda_error(cudaSetDevice(device_id_));
    while (running_) {
        ModelInstanceInferenceTask tasks;
        if (inference_queue_.wait_dequeue_timed(tasks, 10000)) {  // 10ms
            DoInference(std::move(tasks));
        }
    }
}
void ModelServiceInstance::RunInferenceCallback()
{
    while (running_) {
        GenerateCallbackTask task;
        if (inference_callback_queue_.wait_dequeue_timed(task, 10000)) {
            task.callback(std::move(task.response));
        }
    }
}

void ModelServiceInstance::DoInference(ModelInstanceInferenceTask&& task)
{
    std::unordered_map<std::string, ft::Tensor> input_tensors;
    std::unordered_map<std::string, ft::Tensor> output_tensors;
    PrepareInferenceTensors(task, input_tensors, output_tensors);
    if (task.stream) {
        task.stream_gen_lengths = task.start_lengths;
    }
    int rc = DoInference(input_tensors, output_tensors, task.stream);
    if (0 != rc) {}
    else {
    }
    TaskCallback(task, true);
    if (task.done) {
        task.done();
    }
}

}  // namespace ft_ext