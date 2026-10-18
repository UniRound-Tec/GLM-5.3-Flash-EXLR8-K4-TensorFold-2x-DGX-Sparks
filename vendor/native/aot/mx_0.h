
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} mx_0_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_mx_0_cuda_init(void **);
void _mlir_mx_0_cuda_load_to_device(void **);
static inline void mx_0_Kernel_Module_Load(mx_0_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_mx_0_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_mx_0_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void mx_0_Kernel_Module_Unload(mx_0_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void _mlir_mx_0__mlir_ciface_cutlass___call___b12x_libquantmxfp8_rows_MXFP8RowsQuantLaunch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_1_1_1_CUstream0x0(void **args, int32_t num_args);

static inline int32_t cute_dsl_mx_0_wrapper(mx_0_Kernel_Module_t *module, void *source_ptr, void *values_ptr, void *scale_rows_ptr, void *scale_mma_ptr, int32_t m, int32_t source_k, int32_t grid_x, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[9] = {
        &source_ptr, &values_ptr, &scale_rows_ptr, &scale_mma_ptr, &m, &source_k, &grid_x, &stream,
        &ret
    };
    _mlir_mx_0__mlir_ciface_cutlass___call___b12x_libquantmxfp8_rows_MXFP8RowsQuantLaunch_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_1_1_1_CUstream0x0(args, 9);
    return ret;
}
