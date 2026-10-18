// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define M_TILE 64
#define GROUP_SIZE 16
#define PQ2_KS 128
#define PQ2_KP 64
#define PQ2_AP 80
__device__ __forceinline__ void moe_cp_async_pred_16(void* dst,const void* src,bool pred) {
    unsigned int d=__cvta_generic_to_shared(dst),n=pred?16:0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(d),"l"(src),"r"(n));
}
__device__ __forceinline__ void moe_cp_async_pred_8(void* dst,const void* src,bool pred) {
    unsigned int d=__cvta_generic_to_shared(dst),n=pred?8:0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"::"r"(d),"l"(src),"r"(n));
}
__device__ __forceinline__ void moe_cp_async_commit() {asm volatile("cp.async.commit_group;");}
__device__ __forceinline__ float rcp_ftz(float x) {float y;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x));return y;}
__device__ __forceinline__ unsigned char qad_group_scale(float mx,float gs,float* inv) {
    float s=fminf(gs*(mx*rcp_ftz(6.0f)),448.0f);
    unsigned short bits;asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%1;":"=h"(bits):"f"(s));
    __nv_fp8_e4m3 v;v.__x=bits&255;
    float decoded=(float)v;
    *inv=decoded ? rcp_ftz(decoded)*gs : 0.0f;
    return bits&255;
}
__device__ __forceinline__ unsigned int qad_fp4(float x) {
    unsigned int bits;asm("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b,%1,%1; mov.b32 %0,{b,b,b,b};}":"=r"(bits):"f"(x));return bits&15;
}
__device__ __forceinline__ float silu_nvfp4_act(float g,float u) {
    g=fminf(g,10.0f);u=fminf(fmaxf(u,-10.0f),10.0f);
    float sig=rcp_ftz(1.0f+__expf(-g));
    return __bfloat162float(__float2bfloat16((g*sig)*u));
}
// Logical [expert,row,group] -> four exact bytes in existing CSF storage.
__device__ __forceinline__ unsigned int csf_word(const unsigned char* base,unsigned e,
        unsigned rows,unsigned cols,unsigned row,unsigned col) {
    const auto* p=base+1024;
    const auto* words=(const unsigned int*)p;
    unsigned long long slab=(unsigned long long)e*(rows/128)+row/128;
    unsigned long long fixed=slab*128*(1+cols/2);
    unsigned lane=(row%32)*4+(row%128)/32;
    unsigned packed=*(const unsigned short*)(p+fixed+128+(col/4*128+lane)*2);
    unsigned value=(packed|(packed<<8))&0x00ff00ff;
    value=(value|(value<<4))&0x0f0f0f0f;
    value+=(unsigned)p[fixed+lane]*0x01010101;
    unsigned long long tile=slab*(cols/4)+col/4;
    unsigned long long partitions=(unsigned long long)288*rows*(1+cols/2)/4;
    unsigned long long meta=partitions+tile*8;
    unsigned mask=words[meta+2+lane/32],flag=1u<<(lane%32);
    if(mask&flag) {
        unsigned rank=((words[meta+1]>>((lane/32)*8))&255)+__popc(mask&(flag-1));
        unsigned long long records=partitions+(unsigned long long)288*(rows*cols/512)*8;
        value=words[records+words[meta]+rank];
    }
    return value;
}
