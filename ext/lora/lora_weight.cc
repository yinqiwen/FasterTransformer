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
#include "lora_weight.h"

namespace ft_ext {
template<typename T>
LoraWeight<T>::LoraWeight(size_t m, size_t n, size_t k, float alpha, bool transpose):
    m_(m), n_(n), k_(k), alpha_(alpha), transpose_(transpose)
{
}
template<typename T>
LoraWeight<T>::~LoraWeight()
{
    ft::deviceFree(lora_A_);
    ft::deviceFree(lora_B_);
}
template<typename T>
void LoraWeight<T>::load(const std::string& lora_a_file,
                         const std::string& lora_b_file,
                         ft::FtCudaDataType model_file_type)
{
    ft::deviceMalloc(&lora_A_, k_ * n_, false);
    ft::deviceMalloc(&lora_B_, m_ * k_, false);
    ft::loadWeightFromBin(lora_A_, {k_, n_}, lora_a_file, model_file_type);
    ft::loadWeightFromBin(lora_B_, {m_, k_}, lora_b_file, model_file_type);
}

template<typename T>
void LoraWeight<T>::apply(ft::cublasMMWrapper* wrapper, T* weight)
{
    float beta = 1.0;
    if (transpose_) {
        // weight = 1.0*weight+ transpose(B*A)*alpha
        wrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_T, m_, n_, k_, lora_B_, k_, lora_A_, n_, weight, m_, alpha_, beta);
    }
    else {
        // weight = 1.0*weight+ B*A*alpha
        wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n_, m_, k_, lora_A_, n_, lora_B_, k_, weight, n_, alpha_, beta);
    }
}
template<typename T>
void LoraWeight<T>::unapply(ft::cublasMMWrapper* wrapper, T* weight)
{
    float beta = 1.0;
    if (transpose_) {
        wrapper->Gemm(CUBLAS_OP_T, CUBLAS_OP_T, m_, n_, k_, lora_B_, k_, lora_A_, n_, weight, m_, -alpha_, beta);
    }
    else {
        wrapper->Gemm(CUBLAS_OP_N, CUBLAS_OP_N, n_, m_, k_, lora_A_, n_, lora_B_, k_, weight, n_, -alpha_, beta);
    }
}

template class LoraWeight<float>;
template class LoraWeight<half>;
#ifdef ENABLE_BF16
template class LoraWeight<__nv_bfloat16>;
#endif

}  // namespace ft_ext