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
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "3rdparty/INIReader.h"
#include "blockingconcurrentqueue.h"
#include "ext/config/model_config.h"

#include "model.h"
#include "model_service_instance.h"
#include "model_task.h"

namespace ft_ext {

class ModelService {
public:
    bool Generate(const GenerateRequest& req, GenerateCallback&& cb);
    int  Init(const std::string& config_path);
    virtual ~ModelService();

protected:
    virtual int                     OnInit()                                                              = 0;
    virtual ModelPtr                NewModel()                                                            = 0;
    virtual ModelServiceInstancePtr NewModelInstance(int                                     device_id,
                                                     const Model*                            model,
                                                     const ModelServiceConfig&               config,
                                                     ft::NcclParam                           tensor_para,
                                                     ft::NcclParam                           pipeline_para,
                                                     std::shared_ptr<ft::AbstractCustomComm> custom_comm) = 0;

    ModelServiceInstancePtrCluster CreateModelServiceInstanceCluster();

    void Encode(const std::vector<std::string>& txts,
                std::vector<int>&               start_ids,
                std::vector<int>&               start_lengths,
                int&                            max_input_len);

    INIReader          config_reader_;
    ModelServiceConfig config_;

    ModelPtr model_;

private:
    void Start();
    void Run();
    void ProcessGenerateTasks(std::deque<GenerateTask>& tasks);
    void ProcessSingleBatch(ModelServiceInstancePtrCluster& cluster, ModelInstanceInferenceTask& tasks);

    moodycamel::BlockingConcurrentQueue<ModelServiceInstancePtrCluster> instance_queue_;

    moodycamel::BlockingConcurrentQueue<GenerateTask> gen_task_queue_;
    std::unique_ptr<std::thread>                      gen_task_thread_;
    std::mutex                                        gen_task_mutex_;
    std::condition_variable                           gen_task_running_cv_;
    bool                                              running_ = false;
};
using ModelServicePtr     = std::unique_ptr<ModelService>;
using ModelServiceCreator = std::function<ModelServicePtr()>;

struct ModelServiceRegistry {
    ModelServiceRegistry(const std::string& type, const std::string& data_type, const ModelServiceCreator& creator);
};

class ModelServiceFactory {
public:
    static ModelServicePtr NewModelService(const std::string& config_path);
    static void            RegisterModelServiceCreator(const std::string&         type,
                                                       const std::string&         data_type,
                                                       const ModelServiceCreator& creator);
};

}  // namespace ft_ext
#define FT_EXT_CAT(x, y) x##y
#define FT_EXT_CAT2(x, y) FT_EXT_CAT(x, y)
#define FT_EXT_STRINGIZE(s) FT_EXT_STR(s)
#define FT_EXT_STR(s) #s
#define FT_EXT_MODEL_REG(MODEL_TYPE, DATA_TYPE, TYPE)                                                                  \
    static ft_ext::ModelServiceRegistry FT_EXT_CAT2(model, __COUNTER__)(                                               \
        FT_EXT_STRINGIZE(MODEL_TYPE), FT_EXT_STRINGIZE(DATA_TYPE), []() -> ft_ext::ModelServicePtr {                   \
            ft_ext::ModelServicePtr p(new TYPE);                                                                       \
            return p;                                                                                                  \
        });
