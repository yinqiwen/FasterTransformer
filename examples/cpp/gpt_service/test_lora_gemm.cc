#include <memory>
#include <mutex>
#include <vector>

#include "ext/lora/lora_weight.h"
#include "ext/utils/file_utils.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/memory_utils.h"
namespace ft = fastertransformer;

static void test(std::unique_ptr<ft::cublasMMWrapper>& cublas_wrapper)
{
    std::vector<float> lora_B{1, 2, 2, 1, 2, 3};                    // 2*3
    std::vector<float> lora_A{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};  // 3*4
    std::vector<float> weight{10, 10, 10, 10, 20, 20, 20, 20};
    float              alpha = 0.2;
    float              beta  = 1.0;

    float* d_lora_B = nullptr;
    float* d_lora_A = nullptr;
    float* d_weight = nullptr;
    ft::deviceMalloc(&d_lora_B, lora_B.size(), false);
    ft::deviceMalloc(&d_lora_A, lora_A.size(), false);
    ft::deviceMalloc(&d_weight, weight.size(), false);
    ft::cudaH2Dcpy(d_lora_B, lora_B.data(), lora_B.size());
    ft::cudaH2Dcpy(d_lora_A, lora_A.data(), lora_A.size());
    ft::cudaH2Dcpy(d_weight, weight.data(), weight.size());

    cublas_wrapper->setGemmConfig(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 3, d_lora_A, 4, d_lora_B, 3, d_weight, 4, alpha, beta);
    float h_weight[weight.size()];
    ft::cudaD2Hcpy(h_weight, d_weight, weight.size());
    std::string h_weight_str = ft::arr2str(h_weight, weight.size());
    printf("h_weight_str:%s\n", h_weight_str.c_str());

    alpha = -0.2;
    cublas_wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 3, d_lora_A, 4, d_lora_B, 3, d_weight, 4, alpha, beta);
    ft::cudaD2Hcpy(h_weight, d_weight, weight.size());
    h_weight_str = ft::arr2str(h_weight, weight.size());
    printf("h_weight_str:%s\n", h_weight_str.c_str());
}

static void test2(std::unique_ptr<ft::cublasMMWrapper>& cublas_wrapper)
{
    std::vector<float> lora_B{1, 2, 2, 1, 2, 3};                    // 2*3
    std::vector<float> lora_A{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};  // 3*4
    std::vector<float> weight{10, 20, 10, 20, 10, 20, 10, 20};      // 4*2
    float              alpha = 0.2;
    float              beta  = 1.0;

    float* d_lora_B = nullptr;
    float* d_lora_A = nullptr;
    float* d_weight = nullptr;
    ft::deviceMalloc(&d_lora_B, lora_B.size(), false);
    ft::deviceMalloc(&d_lora_A, lora_A.size(), false);
    ft::deviceMalloc(&d_weight, weight.size(), false);
    ft::cudaH2Dcpy(d_lora_B, lora_B.data(), lora_B.size());
    ft::cudaH2Dcpy(d_lora_A, lora_A.data(), lora_A.size());
    ft::cudaH2Dcpy(d_weight, weight.data(), weight.size());

    cublas_wrapper->setGemmConfig(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
    cublas_wrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_T, 2, 4, 3, d_lora_B, 3, d_lora_A, 4, d_weight, 2, alpha, beta);
    float h_weight[weight.size()];
    ft::cudaD2Hcpy(h_weight, d_weight, weight.size());
    std::string h_weight_str = ft::arr2str(h_weight, weight.size());
    printf("h_weight_str:%s\n", h_weight_str.c_str());
    alpha = -0.2;
    cublas_wrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_T, 2, 4, 3, d_lora_B, 3, d_lora_A, 4, d_weight, 2, alpha, beta);
    ft::cudaD2Hcpy(h_weight, d_weight, weight.size());
    h_weight_str = ft::arr2str(h_weight, weight.size());
    printf("h_weight_str:%s\n", h_weight_str.c_str());
}

static void test_lora1(std::unique_ptr<ft::cublasMMWrapper>& cublas_wrapper)
{
    std::string lora_path   = "/data/DeepDist/models/lora3.6B/1-gpu/test_lora/";
    std::string lora_a_file = "h.0.attn.c_attn.lora_A.weight.bin";
    std::string lora_b_file = "h.0.attn.c_attn.lora_B.weight.bin";
    std::string orig_weight_file =
        "/data/DeepDist/models/lora3.6B/1-gpu/model.layers.0.attention.query_key_value.weight.0.bin";
    std::string cmp_weight_file =
        "/data/DeepDist/models/lora3.6B/lora/1-gpu/model.layers.0.attention.query_key_value.weight.0.bin";
    int                      m = 9216;
    int                      n = 3072;
    int                      k = 16;
    ft_ext::LoraWeight<half> lora_w(m, n, k, 4, false);
    lora_w.load(lora_path + lora_a_file, lora_path + lora_b_file, ft::FP16);

    half* d_orig_w = nullptr;
    ft::deviceMalloc(&d_orig_w, m * n, false);
    ft::loadWeightFromBin(d_orig_w, {m, n}, orig_weight_file, ft::FP16);
    cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F);
    lora_w.apply(cublas_wrapper.get(), d_orig_w);

    std::string applied_file =
        "/data/DeepDist/models/lora3.6B/1-gpu/merge_lora/model.layers.0.attention.query_key_value.weight.0.bin";
    std::vector<half> host_merged(m * n);
    ft::cudaD2Hcpy(&host_merged[0], d_orig_w, host_merged.size());
    ft_ext::file_write_content(applied_file, host_merged.data(), host_merged.size() * sizeof(half));

    std::vector<half> cmp_merged(m * n);
    ft_ext::file_read_content(cmp_weight_file, &cmp_merged[0], cmp_merged.size() * sizeof(half));

    int diff_count = 0;
    for (int i = 0; i < cmp_merged.size(); i++) {
        float x = (float)cmp_merged[i];
        float y = (float)host_merged[i];
        if (x != y) {
            diff_count++;
        }
    }
    FT_LOG_INFO("diff_count: %d, total:%d ", diff_count, host_merged.size());
}

int main()
{
    int                                                     device_id = 0;
    cudaStream_t                                            stream_;
    cublasHandle_t                                          cublas_handle_;
    cublasLtHandle_t                                        cublaslt_handle_;
    std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper_;
    std::mutex                                              cublas_wrapper_mutex_;
    std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    allocator_ = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(device_id);

    cudaStreamCreate(&stream_);

    allocator_->setStream(stream_);
    cublasCreate(&cublas_handle_);
    cublasLtCreate(&cublaslt_handle_);
    cublasSetStream(cublas_handle_, stream_);
    std::unique_ptr<ft::cublasAlgoMap> cublas_algo_map_;
    cublas_algo_map_ = std::make_unique<ft::cublasAlgoMap>(GEMM_CONFIG);
    cublas_wrapper_  = std::make_unique<ft::cublasMMWrapper>(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(), &cublas_wrapper_mutex_, allocator_.get());

    // test(cublas_wrapper_);
    // test2(cublas_wrapper_);
    test_lora1(cublas_wrapper_);

    cublas_wrapper_.reset();
    return 0;
}