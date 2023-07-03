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
#include "model_service.h"
#include <algorithm>
#include <unordered_map>

#include "src/fastertransformer/utils/nccl_utils.h"

#include "ext/utils/file_utils.h"
#include "ext/utils/spdlogger.h"
#include "ext/utils/time_utils.h"

namespace ft_ext {

ModelServiceRegistry::ModelServiceRegistry(const std::string&         type,
                                           const std::string&         data_type,
                                           const ModelServiceCreator& creator)
{
    ModelServiceFactory::RegisterModelServiceCreator(type, data_type, creator);
}

using ModelServiceCreatorTable = std::unordered_map<std::string, ModelServiceCreator>;
static std::unique_ptr<ModelServiceCreatorTable> g_modle_service_creators = nullptr;
ModelServicePtr                                  ModelServiceFactory::NewModelService(const std::string& config_path)
{
    if (!g_modle_service_creators) {
        FT_ERROR("Empty model service creator registry!");
        return nullptr;
    }
    INIReader reader = INIReader(config_path);
    if (reader.ParseError() < 0) {
        FT_ERROR("Can't load {}", config_path);
        return nullptr;
    }
    std::string model_type          = reader.Get("ft_instance_hyperparameter", "model_type");
    std::string data_type           = reader.Get("ft_instance_hyperparameter", "data_type");
    std::string type_with_data_type = model_type + ":" + data_type;
    auto        found               = g_modle_service_creators->find(type_with_data_type);
    if (found == g_modle_service_creators->end()) {
        FT_ERROR("No creator found for model by key:{}", type_with_data_type);
        return nullptr;
    }

    auto service = found->second();
    if (!service) {
        FT_ERROR("Create null service by key:{}", type_with_data_type);
        return nullptr;
    }
    int rc = service->Init(config_path);
    if (0 != rc) {
        return nullptr;
    }
    return service;
}

void ModelServiceFactory::RegisterModelServiceCreator(const std::string&         type,
                                                      const std::string&         data_type,
                                                      const ModelServiceCreator& creator)
{
    if (nullptr == g_modle_service_creators) {
        g_modle_service_creators = std::make_unique<ModelServiceCreatorTable>();
    }
    std::string type_with_data_type = type + ":" + data_type;
    g_modle_service_creators->emplace(type_with_data_type, creator);
}

static std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> CreateNcclParams(const int tensor_para_size,
                                                                                          const int pipeline_para_size,
                                                                                          const int node_id,
                                                                                          const int device_id_start = 0,
                                                                                          const bool multi_node = false)
{
    const int gpu_count       = ft::getDeviceCount();
    const int local_comm_size = multi_node ? gpu_count : tensor_para_size * pipeline_para_size;
    ft::FT_CHECK(tensor_para_size > 0 && pipeline_para_size > 0);
    ft::FT_CHECK(device_id_start + (int)local_comm_size <= gpu_count);

    std::vector<ft::NcclUid> nccl_ids;
    if (tensor_para_size > 1 || pipeline_para_size > 1) {
        nccl_ids.resize(tensor_para_size + pipeline_para_size);
        if (node_id == 0) {
            for (uint32_t i = 0; i < nccl_ids.size(); i++) {
                ft::ftNcclGetUniqueId(nccl_ids[i]);
            }
        }
        for (size_t i = 0; i < nccl_ids.size(); i++) {
            ft::mpi::bcast(&nccl_ids[i], sizeof(nccl_ids[i]), ft::mpi::MPI_TYPE_BYTE, 0, ft::mpi::COMM_WORLD);
        }
    }

    std::vector<ft::NcclParam> tensor_para_params(local_comm_size);
    std::vector<ft::NcclParam> pipeline_para_params(local_comm_size);
    // Don't init comm when size == 1
    if (tensor_para_size > 1) {
        ft::ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            ft::NcclUid tensor_para_nccl_uid = nccl_ids[pipeline_para_rank];
            ft::check_cuda_error(cudaSetDevice(gid));
            ft::ftNcclCommInitRank(
                tensor_para_params[gid - device_id_start], tensor_para_rank, tensor_para_size, tensor_para_nccl_uid);
        }
        ft::ftNcclGroupEnd();
    }
    if (pipeline_para_size > 1) {
        ft::ftNcclGroupStart();
        for (int gid = device_id_start; gid < device_id_start + local_comm_size; gid++) {
            int rank               = node_id * gpu_count + gid - device_id_start;
            int tensor_para_rank   = rank % tensor_para_size;
            int pipeline_para_rank = rank / tensor_para_size;

            ft::NcclUid pipeline_para_nccl_uid = nccl_ids[pipeline_para_size + tensor_para_rank];
            ft::check_cuda_error(cudaSetDevice(gid));
            ft::ftNcclCommInitRank(pipeline_para_params[gid - device_id_start],
                                   pipeline_para_rank,
                                   pipeline_para_size,
                                   pipeline_para_nccl_uid);
        }
        ft::ftNcclGroupEnd();
    }
    return std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>(tensor_para_params, pipeline_para_params);
}

static void CreateCustomComms(std::vector<std::shared_ptr<ft::AbstractCustomComm>>* custom_all_reduce_comms,
                              const std::string&                                    data_type,
                              int                                                   enable_custom_all_reduce,
                              int                                                   world_size)
{
    if (data_type == "fp32") {
        using commDataType = typename ft::CustomARCommTypeConverter<float>::Type;
        ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce, world_size);
    }
    else if (data_type == "fp16") {
        using commDataType = typename ft::CustomARCommTypeConverter<half>::Type;
        ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce, world_size);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        using commDataType = typename ft::CustomARCommTypeConverter<__nv_bfloat16>::Type;
        ft::initCustomAllReduceComm<commDataType>(custom_all_reduce_comms, enable_custom_all_reduce, world_size);
    }
#endif
    else {
        FT_ERROR("Can NOT create custom comms by data_type:{}", data_type);
    }
}

ModelService::~ModelService()
{
    running_ = false;
    if (gen_task_thread_) {
        gen_task_thread_->join();
    }
}

ModelServiceInstancePtrCluster ModelService::CreateModelServiceInstanceCluster()
{
    ModelServiceInstancePtrCluster cluster;
    int                            world_size = config_.model.tensor_para_size * config_.model.pipeline_para_size;
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params =
        CreateNcclParams(config_.model.tensor_para_size, config_.model.pipeline_para_size, 0, 0);
    std::vector<std::shared_ptr<ft::AbstractCustomComm>> custom_all_reduce_comms;
    CreateCustomComms(
        &custom_all_reduce_comms, config_.model.data_type, config_.model.enable_custom_all_reduce, world_size);
    int device_id = 0;
    for (int tensor_rank = 0; tensor_rank < config_.model.tensor_para_size; tensor_rank++) {
        for (int pipeline_rank = 0; pipeline_rank < config_.model.pipeline_para_size; pipeline_rank++) {
            const int     comms_rank  = device_id % (config_.model.tensor_para_size * config_.model.pipeline_para_size);
            ft::NcclParam tensor_para = nccl_params.first[comms_rank];
            ft::NcclParam pipeline_para      = nccl_params.second[comms_rank];
            ModelServiceInstancePtr instance = NewModelInstance(
                device_id, model_.get(), config_, tensor_para, pipeline_para, custom_all_reduce_comms[comms_rank]);
            cluster.emplace_back(instance);
            device_id++;
        }
    }
    return cluster;
}

int ModelService::Init(const std::string& config_path)
{
    INIReader reader = INIReader(config_path);
    if (reader.ParseError() < 0) {
        FT_ERROR("Can't load {}", config_path);
        return -1;
    }
    config_reader_ = reader;
    int rc         = config_.Load(config_reader_);
    if (0 != rc) {
        FT_ERROR("Invalid config:{}", rc);
        return rc;
    }

    model_ = NewModel();
    if (config_.tokenizer == "bert") {
        std::string vocab_path = config_.model.model_dir + "/vocab.txt";
        if (!is_file_exist(vocab_path)) {
            FT_ERROR("{} is not exist.", vocab_path);
            return -1;
        }
        std::string vocab_content;
        file_read_all(vocab_path, vocab_content);
        model_->tokenizer         = std::make_unique<BertTokenizer>(vocab_content);
        model_->tokenizer_decoder = std::make_unique<BertTokenizerDecoder>(vocab_content);
    }
    else {
        FT_ERROR("Unsupported tokenizer:{}", config_.tokenizer);
        return -1;
    }

    FT_INFO("Model dir:{}, data_type:{}, int8_mode:{}, hidden_units:{},inter_size:{}, vocab_size:{}",
            config_.model.model_dir,
            config_.model.data_type,
            config_.model.int8_mode,
            config_.model.hidden_units,
            config_.model.inter_size,
            config_.model.vocab_size);

    rc = OnInit();
    if (0 != rc) {
        FT_ERROR("Failed to init model service with rc:{}", rc);
    }
    else {
        FT_INFO("Success to init model service.");
    }

    for (size_t i = 0; i < config_.worker_instance_num; i++) {
        ModelServiceInstancePtrCluster cluster = CreateModelServiceInstanceCluster();
        instance_queue_.try_enqueue(std::move(cluster));
    }

    Start();
    return rc;
}

void ModelService::Start()
{
    gen_task_thread_ = std::make_unique<std::thread>(&ModelService::Run, this);
    {
        std::unique_lock lk(gen_task_mutex_);
        gen_task_running_cv_.wait(lk, [this] { return running_; });
    }
}

void ModelService::Run()
{
    {
        std::lock_guard lk(gen_task_mutex_);
        running_ = true;
    }
    gen_task_running_cv_.notify_one();
    std::deque<GenerateTask> sorted_tasks;
    while (running_) {
        if (sorted_tasks.size() <= 4) {
            std::vector<GenerateTask> ready_tasks(128);
            size_t                    ready_n = gen_task_queue_.wait_dequeue_bulk_timed(
                ready_tasks.begin(), ready_tasks.size(), config_.task_queue_wait_usecs);
            ready_tasks.resize(ready_n);
            if (ready_n > 0) {
                sorted_tasks.insert(sorted_tasks.end(), ready_tasks.begin(), ready_tasks.end());
                // std::sort(sorted_tasks.begin(), sorted_tasks.end(), customLess);
            }
        }
        ProcessGenerateTasks(sorted_tasks);
    }
}

void ModelService::ProcessGenerateTasks(std::deque<GenerateTask>& tasks)
{
    int64_t enter_us = gettimeofday_us();
    while (!tasks.empty()) {
        ModelServiceInstancePtrCluster cluster;
        if (!instance_queue_.wait_dequeue_timed(cluster, config_.fetch_inference_instance_timeout_usecs)) {
            return;
        }
        int64_t                   fetch_inference_instance_us = gettimeofday_us() - enter_us;
        std::vector<GenerateTask> batch;
        while (batch.size() < config_.max_batch_size && !tasks.empty()) {
            bool should_merge_batch = true;
            if (batch.size() > 0) {
                if (batch[0].request.stream != tasks[0].request.stream) {
                    should_merge_batch = false;
                }
                else if (batch[0].request.beam_width != tasks[0].request.beam_width) {
                    should_merge_batch = false;
                }
            }
            if (should_merge_batch) {
                batch.emplace_back(std::move(tasks[0]));
                tasks.pop_front();
            }
            else {
                break;
            }
        }
        ModelInstanceInferenceTask inference_task;
        for (auto& task : batch) {
            GenerateStat task_stat;
            task_stat.start_us                    = task.request.start_us;
            task_stat.batch_queue_wait_us         = enter_us - task.request.start_us;
            task_stat.fetch_inference_instance_us = fetch_inference_instance_us;
            inference_task.stats.emplace_back(task_stat);
        }
        inference_task.batch = std::move(batch);
        ProcessSingleBatch(cluster, inference_task);
    }
}

void ModelService::ProcessSingleBatch(ModelServiceInstancePtrCluster& cluster,
                                      ModelInstanceInferenceTask&     baseline_inference_task)
{
    auto& tasks                    = baseline_inference_task.batch;
    baseline_inference_task.stream = tasks[0].request.stream;
    baseline_inference_task.max_request_output_len =
        std::max_element(tasks.begin(), tasks.end(), [](const auto& a, const auto& b) {
            return a.request.request_output_len < b.request.request_output_len;
        })->request.request_output_len;
    if (tasks[0].request.beam_width > 1) {
        bool same_beam_search_diversity_rate = true;
        for (auto& task : tasks) {
            baseline_inference_task.beam_search_diversity_rate.emplace_back(task.request.beam_search_diversity_rate);
            if (same_beam_search_diversity_rate
                && baseline_inference_task.beam_search_diversity_rate[0] != task.request.beam_search_diversity_rate) {
                same_beam_search_diversity_rate = false;
            }
        }
        if (same_beam_search_diversity_rate) {
            baseline_inference_task.beam_search_diversity_rate.resize(1);
        }
    }
    else {
        bool smae_top_k = true;
        bool same_top_p = true;
        for (auto& task : tasks) {
            baseline_inference_task.runtime_top_k.emplace_back(task.request.top_k);
            baseline_inference_task.runtime_top_p.emplace_back(task.request.top_p);

            if (smae_top_k && baseline_inference_task.runtime_top_k[0] != task.request.top_k) {
                smae_top_k = false;
            }
            if (same_top_p && baseline_inference_task.runtime_top_p[0] != task.request.top_p) {
                same_top_p = false;
            }
        }
        if (smae_top_k) {
            baseline_inference_task.runtime_top_k.resize(1);
        }
        if (same_top_p) {
            baseline_inference_task.runtime_top_p.resize(1);
        }
    }
    bool smae_temperature        = true;
    bool same_len_penalty        = true;
    bool smae_repetition_penalty = true;
    bool same_presence_penalty   = true;
    bool same_min_length         = true;
    for (auto& task : tasks) {
        GenerateStat task_stat;
        task_stat.start_us            = task.request.start_us;
        task_stat.batch_queue_wait_us = gettimeofday_us() - task.request.start_us;
        baseline_inference_task.stats.emplace_back(task_stat);

        baseline_inference_task.output_seq_len.emplace_back(task.request.request_output_len
                                                            + baseline_inference_task.max_input_len);
        baseline_inference_task.temperature.emplace_back(task.request.temperature);
        baseline_inference_task.len_penalty.emplace_back(task.request.len_penalty);
        baseline_inference_task.repetition_penalty.emplace_back(task.request.repetition_penalty);
        baseline_inference_task.presence_penalty.emplace_back(task.request.presence_penalty);
        baseline_inference_task.min_length.emplace_back(task.request.min_length);

        if (smae_temperature && baseline_inference_task.temperature[0] != task.request.temperature) {
            smae_temperature = false;
        }
        if (same_len_penalty && baseline_inference_task.len_penalty[0] != task.request.len_penalty) {
            same_len_penalty = false;
        }
        if (smae_repetition_penalty
            && baseline_inference_task.repetition_penalty[0] != task.request.repetition_penalty) {
            smae_repetition_penalty = false;
        }
        if (same_presence_penalty && baseline_inference_task.presence_penalty[0] != task.request.presence_penalty) {
            same_presence_penalty = false;
        }
        if (same_min_length && baseline_inference_task.min_length[0] != task.request.min_length) {
            same_min_length = false;
        }
    }
    if (same_min_length) {
        baseline_inference_task.min_length.resize(1);
    }
    if (smae_temperature) {
        baseline_inference_task.temperature.resize(1);
    }
    if (same_len_penalty) {
        baseline_inference_task.len_penalty.resize(1);
    }
    if (smae_repetition_penalty) {
        baseline_inference_task.repetition_penalty.resize(1);
        if (baseline_inference_task.repetition_penalty[0] == 1.0) {
            baseline_inference_task.repetition_penalty.clear();
        }
    }
    if (same_presence_penalty) {
        baseline_inference_task.presence_penalty.resize(1);
        if (baseline_inference_task.presence_penalty[0] == 0.0) {
            baseline_inference_task.presence_penalty.clear();
        }
    }

    std::vector<std::string> txts;
    for (auto& task : tasks) {
        txts.emplace_back(task.request.prompt);
    }
    Encode(txts,
           baseline_inference_task.start_ids,
           baseline_inference_task.start_lengths,
           baseline_inference_task.max_input_len);
    baseline_inference_task.done = [cluster, this]() {
        bool r = instance_queue_.try_enqueue(std::move(cluster));
        if (!r) {
            FT_ERROR("Failed to recycle inference instance.");
        }
        else {
            FT_DEBUG("Success to recycle inference instance wtih size:{}", instance_queue_.size_approx());
        }
    };
    for (size_t i = 0; i < cluster.size() - 1; i++) {
        cluster[i]->Post(baseline_inference_task);
        baseline_inference_task.done = {};
    }
    cluster[cluster.size() - 1]->Post(std::move(baseline_inference_task));
}

void ModelService::Encode(const std::vector<std::string>& txts,
                          std::vector<int>&               start_ids,
                          std::vector<int>&               start_lengths,
                          int&                            max_input_len)
{
    std::vector<std::vector<int>> tmp_start_ids;
    for (const auto& txt : txts) {
        auto ids = model_->tokenizer->Encode(model_->tokenizer->Tokenize(ustring(txt)));
        ids      = model_->tokenizer->AddSpecialToken(ids);
        ids.pop_back();
        tmp_start_ids.push_back(ids);
        start_lengths.push_back(ids.size());
    }

    max_input_len = start_lengths[0];
    for (uint i = 1; i < (uint)start_lengths.size(); i++) {
        max_input_len = max_input_len > start_lengths[i] ? max_input_len : start_lengths[i];
    }

    // Add padding
    for (int i = 0; i < (int)tmp_start_ids.size(); i++) {
        for (int j = (int)tmp_start_ids[i].size(); j < max_input_len; j++) {
            tmp_start_ids[i].push_back(config_.model.end_id);
        }
    }
    for (auto& ids : tmp_start_ids) {
        start_ids.insert(start_ids.end(), ids.begin(), ids.end());
    }
}

bool ModelService::Generate(const GenerateRequest& req, GenerateCallback&& cb)
{
    GenerateTask task;
    task.request          = req;
    task.callback         = std::move(cb);
    task.request.start_us = gettimeofday_us();
    bool r                = gen_task_queue_.try_enqueue(std::move(task));
    if (!r) {
        FT_ERROR("Failed to enqueue generate task with queue appromix size:{}", gen_task_queue_.size_approx());
    }
    return r;
}

}  // namespace ft_ext