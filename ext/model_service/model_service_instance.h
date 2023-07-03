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

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ext/config/model_config.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cublasAlgoMap.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include "blockingconcurrentqueue.h"

#include "model.h"
#include "model_task.h"

namespace ft_ext {
namespace ft = fastertransformer;
class ModelServiceInstance {
public:
    ModelServiceInstance(int                                     device_id,
                         const Model*                            model,
                         const ModelServiceConfig&               config,
                         ft::NcclParam                           tensor_para,
                         ft::NcclParam                           pipeline_para,
                         std::shared_ptr<ft::AbstractCustomComm> custom_comm);
    bool Post(const ModelInstanceInferenceTask& task);
    bool Post(ModelInstanceInferenceTask&& task);

    void StreamingGenerateCallback(ModelInstanceInferenceTask& task);

    virtual ~ModelServiceInstance();

protected:
    virtual int  PrepareInferenceTensors(ModelInstanceInferenceTask&                  tasks,
                                         std::unordered_map<std::string, ft::Tensor>& input,
                                         std::unordered_map<std::string, ft::Tensor>& output) = 0;
    virtual int  DoInference(std::unordered_map<std::string, ft::Tensor>& input,
                             std::unordered_map<std::string, ft::Tensor>& output,
                             bool                                         stream)                                                     = 0;
    virtual void GetOutput(size_t                         request_batch_size,
                           size_t                         beam_width,
                           size_t                         max_input_len,
                           size_t                         max_request_output_len,
                           const std::vector<int>&        ignore_prefix_lengths,
                           std::vector<std::vector<int>>& output_ids)                         = 0;

    void DoInference(ModelInstanceInferenceTask&& tasks);
    bool PostCallback(const GenerateCallback& cb, GenerateResponse&& response);

    int                                     device_id_;
    const Model*                            model_;
    ModelServiceConfig                      config_;
    ft::NcclParam                           tensor_para_;
    ft::NcclParam                           pipeline_para_;
    std::shared_ptr<ft::AbstractCustomComm> custom_comm_;

    cudaDeviceProp device_prop_;

    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    cudaStream_t                                            stream_;
    cublasHandle_t                                          cublas_handle_;
    cublasLtHandle_t                                        cublaslt_handle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle_;
#else
    std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map_;
#endif
    std::unique_ptr<ft::cublasMMWrapper> cublas_wrapper_;
    std::mutex                           cublas_wrapper_mutex_;

    uint64_t random_seed_ = 0;

private:
    void RunInference();
    void RunInferenceCallback();

    void FillGenerateResponse(const ModelInstanceInferenceTask&    tasks,
                              const std::vector<std::vector<int>>& output_ids,
                              std::vector<GenerateResponse>&       batch_response);
    void TaskCallback(const ModelInstanceInferenceTask& tasks, bool complete);

    bool                                                            running_ = false;
    std::unique_ptr<std::thread>                                    inference_thread_;
    std::unique_ptr<std::thread>                                    inference_callback_thread_;
    moodycamel::BlockingConcurrentQueue<ModelInstanceInferenceTask> inference_queue_;
    moodycamel::BlockingConcurrentQueue<GenerateCallbackTask>       inference_callback_queue_;
};

using ModelServiceInstancePtr        = std::shared_ptr<ModelServiceInstance>;
using ModelServiceInstancePtrCluster = std::vector<ModelServiceInstancePtr>;
}  // namespace ft_ext