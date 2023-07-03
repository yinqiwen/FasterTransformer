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
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/fastertransformer/utils/Tensor.h"

namespace ft_ext {
enum GenerateErrCode {
    ERR_OK = 0,
};

struct GenerateRequest {
    std::string prompt;

    int   min_length                 = 0;
    int   beam_width                 = 1;
    int   request_output_len         = 128;
    int   top_k                      = 3;
    float top_p                      = 1.0;
    float beam_search_diversity_rate = 0.0;
    float temperature                = 1.0;
    float len_penalty                = 0.0;
    float repetition_penalty         = 1.0;
    float presence_penalty           = 0.0;
    bool  stream                     = false;

    int64_t start_us = 0;
};
struct GenerateStat {
    int64_t start_us                    = 0;
    int64_t batch_queue_wait_us         = 0;
    int64_t fetch_inference_instance_us = 0;
    int64_t complete_us                 = 0;
};
struct GenerateResponse {
    std::vector<std::string> choices;

    GenerateStat stat;
    bool         complete = false;
    int          err_code = 0;
};

using GenerateCallback = std::function<void(GenerateResponse&&)>;

struct GenerateTask {
    GenerateRequest  request;
    GenerateCallback callback;
};
using BatchGenerateTask = std::vector<GenerateTask>;

using ModelInstanceInferenceDone = std::function<void(void)>;

struct ModelInstanceInferenceTask {
    BatchGenerateTask batch;

    bool stream = false;

    std::vector<int> start_lengths;
    std::vector<int> start_ids;
    int              max_input_len          = 0;
    int              max_request_output_len = 0;

    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      input_lengths_h [batch_size] on cpu, optional
    //      prompt_learning_task_name_ids [batch_size] on cpu
    //      output_seq_len [batch_size] on cpu
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_lengths_h [batch_size], cpu, optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      request_prompt_type [batch_size], int, optional
    //      is_return_context_cum_log_probs [1] on cpu, bool, optional
    //      session_len [1] on cpu, uint32, optional
    //      memory_len [1] on cpu, uint32, optional
    //      continue_gen [1] on cpu, bool, optional
    //      is_return_context_embeddings [1] on cpu, bool, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional
    //      repetition_penalty_ignore_orig_input [1] or [batch_size] on cpu, optional

    std::vector<int>      output_seq_len;
    std::vector<int>      start_id;
    std::vector<int>      end_id;
    std::vector<int>      runtime_top_k;
    std::vector<float>    runtime_top_p;
    std::vector<float>    beam_search_diversity_rate;
    std::vector<float>    temperature;
    std::vector<float>    len_penalty;
    std::vector<float>    repetition_penalty;
    std::vector<float>    presence_penalty;
    std::vector<int>      min_length;
    std::vector<uint64_t> random_seed;

    ModelInstanceInferenceDone done;

    std::vector<GenerateStat> stats;
    std::vector<int>          stream_gen_lengths;
};

struct GenerateCallbackTask {
    GenerateResponse response;
    GenerateCallback callback;
};

}  // namespace ft_ext