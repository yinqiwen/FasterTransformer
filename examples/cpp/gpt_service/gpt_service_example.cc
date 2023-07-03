#include <condition_variable>
#include <mutex>

#include "ext/model_service/model_service.h"
#include "ext/utils/time_utils.h"

#include "src/fastertransformer/utils/cuda_utils.h"

using ModelServicePtr = std::unique_ptr<ft_ext::ModelService>;

static void testSingle(ModelServicePtr& gpt_service)
{
    ft_ext::GenerateRequest request;
    request.prompt = "这是很久之前的事情了";
    std::condition_variable cv;
    std::mutex              mutex;
    bool                    done = false;
    for (int i = 0; i < 5; i++) {
        done = false;
        gpt_service->Generate(request, [&](ft_ext::GenerateResponse&& response) {
            for (auto& txt : response.choices) {
                printf("%s", request.prompt.c_str());
                printf("%s\n", txt.c_str());
            }
            size_t free_bytes, total_bytes;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            printf("free_bytes=%llu, total_bytes=%llu\n", free_bytes, total_bytes);
            printf(
                "Cost total_us=%lld,batch_queue_wait_us=%lld,fetch_inference_instance_us=%lld to generate single text.\n",
                response.stat.complete_us - response.stat.start_us,
                response.stat.batch_queue_wait_us,
                response.stat.fetch_inference_instance_us);

            done = true;
            cv.notify_one();
        });
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [&] { return done; });
        }
    }
}
static void testStreaming(ModelServicePtr& gpt_service)
{
    ft_ext::GenerateRequest request;
    request.prompt = "这是很久之前的事情了";
    request.stream = true;
    std::condition_variable cv;
    std::mutex              mutex;
    bool                    done = false;
    printf("%s", request.prompt.c_str());
    gpt_service->Generate(request, [&](ft_ext::GenerateResponse&& response) {
        for (auto& txt : response.choices) {
            printf("%s", txt.c_str());
        }
        if (response.complete) {
            done = true;
            cv.notify_one();
            printf(
                "\nCost total_us=%lld,batch_queue_wait_us=%lld,fetch_inference_instance_us=%lld to generate single text.\n",
                response.stat.complete_us - response.stat.start_us,
                response.stat.batch_queue_wait_us,
                response.stat.fetch_inference_instance_us);
        }
    });
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return done; });
    }
}

// static void testDynamicBatch(ModelServicePtr& gpt_service)
// {
//     std::atomic<uint32_t>    counter{4};
//     std::condition_variable  cv;
//     std::mutex               mutex;
//     bool                     done     = false;
//     int64_t                  start_ms = ft_ext::gettimeofday_ms();
//     std::vector<std::string> txts;
//     txts.push_back("这是很久之前的事情了");
//     txts.push_back("理财是一个阶段性的");
//     txts.push_back("这个向善实践活动");
//     txts.push_back("值得注意的是");
//     for (uint32_t i = 0; i < counter.load(); i++) {
//         gpt_service->Generate(txts[i], [&cv, &counter, &done, start_ms](ft_ext::GenerateResponse&& response) {
//             for (auto& txt : response.results) {
//                 printf("%s\n", txt.c_str());
//             }
//             printf("Cost total_us=%lld,batch_queue_wait_us=%lld to generate single text.\n",
//                    response.stat.complete_us - response.stat.start_us,
//                    response.stat.batch_queue_wait_us);
//             if (counter.fetch_sub(1) == 1) {
//                 int64_t end_ms = ft_ext::gettimeofday_ms();
//                 printf("Cost %lldms to generate batch text.\n", end_ms - start_ms);
//                 done = true;
//                 cv.notify_one();
//             }
//         });
//     }
//     {
//         std::unique_lock<std::mutex> lk(mutex);
//         cv.wait(lk, [&] { return done; });
//     }
// }

int main()
{
    srand(0);
    std::string config_path = "../examples/cpp/gpt_service/gpt_config.ini";
    auto        gpt_service = ft_ext::ModelServiceFactory::NewModelService(config_path);
    testSingle(gpt_service);
    // testSingle(gpt_service);
    // testSingle(gpt_service);
    testStreaming(gpt_service);
    // testDynamicBatch(gpt_service);

    return 0;
}