// GLM-5.3-Flash's EXL3 routed experts on CUDA: the grouped trellis GEMV and the Hadamard rotations around it.
//
// Format (see exl3.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): a 16x16 tile is 32 little-endian
// 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values
// are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile), so a tile goes from memory
// to the tensor cores without a shuffle or a layout change.
//
// Every output depends only on its own row: the rows of a window share an mma tile but mma keeps rows
// independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a
// fixed order (shared memory, then the epilogue kernels), never with atomics.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    __nv_fp8x2_e4m3 fp8(r);
    half2 rounded = static_cast<half2>(fp8);
    return *reinterpret_cast<uint32_t*>(&rounded);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

// decode_tile's values with fewer integer instructions (the prompt kernels): the byte-aligned 16-bit states taken
// with one PRMT each and w >> 4 shared by two states. The same states, so the same bits.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// 16 bytes of trellis words, read once a window (no L1 allocation); asm volatile keeps it where it stands relative
// to the (volatile) mma, so the ring depth is what the code says (patches/0580 of jayleaton/glm53-tensorfold-spark).
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Program (member row, 128-block of K, matrix): Xh[mat][row][block] = fp16((x[row] * suh[mat][e]) @ H) for the
// gate and up projections of every routed slot (row, slot) with slot < slots - 1 (the last slot is the shared
// expert). One warp per program. ROWS (prompt chunks of a layer whose experts all share one suh for gate and up):
// program (row, block) writes Xh[row][block] from suh0 = that vector, the same arithmetic as every pair's input of
// that row, so each pair's fp16 input has the same bits, computed once a row instead of 2 x top_k times.
template <bool ROWS>
__global__ void rot_in_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    int row, e;
    if (ROWS) {
        row = p;
        e = 0;
    } else {
        row = p / slots;
        const int slot = p % slots;
        if (slot == slots - 1) return;
        e = pick[row * slots + slot];
    }
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the rank's intermediate width): gate and up outputs summed over the splits in
// order, rotated, scaled by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down
// projection's input rotation: Xd[row][block] = fp16((act * suh_d[e]) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int slots, float limit) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): the down projection's output summed over the splits in
// order, rotated and scaled by svh: Y[row][slot][:] (fp32, this rank's share of the expert's output).
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int slots) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

template <int NT, int W, int STEPS, int FUSE, bool XROW, int LD = 0, int PD = 1>
__global__ void __launch_bounds__(W * 32) dec8_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(FUSE == 0 || NT * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(LD == 0 || (NT % 4 == 0 && PD >= 1 && STEPS % PD == 0), "LD 1: NT a multiple of 4, PD | STEPS");
    const int item = blockIdx.y, nblock = blockIdx.x;
    int e, first, cnt;
    if constexpr (LD == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = blockIdx.z / SK, split = blockIdx.z % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    constexpr int NV = NT / 4;                           // LD 1: 16-byte loads a lane a k step
    uint4 rv[LD == 1 ? PD : 1][LD == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (LD == 1) {
#pragma unroll
        for (int d = 0; d < PD; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (XROW ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (XROW ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * NT;
    const int kt0 = split * (KT / SK) + warp * STEPS;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (LD == 1) {
        uint32_t ra[PD][4];
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < STEPS; kb += PD) {
#pragma unroll
            for (int d = 0; d < PD; ++d) {
                const int s = kb + d;
                const bool more = s + PD < STEPS;
                uint32_t a[4], words[NT];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + PD) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[NT], a[4];
#pragma unroll
    for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + STEPS; ++kt) {
        uint32_t nw[NT], na[4];
        const bool more = kt + 1 < kt0 + STEPS;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (FUSE == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += W) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * NT * 16 / (W * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (FUSE == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.x + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += W) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}

} // namespace
template <int STEPS>
static void dec_launch(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& T0, const at::Tensor& T1,
                       const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& Z,
                       int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t max_items, int64_t E,
                       int64_t slots, int64_t fuse, bool xrow, const at::Tensor& sv0, const at::Tensor& sv1,
                       const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld) {
    constexpr int NT = 8, W = 4;
    constexpr int PD4 = STEPS % 4 == 0 ? 4 : 2;
    dim3 grid((unsigned)(N / (16 * NT)), (unsigned)max_items, (unsigned)(mats * SK));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto h = [](const at::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); };
    auto w = [](const at::Tensor& t) { return reinterpret_cast<const uint32_t*>(t.data_ptr()); };
#define DEC1(F_, XR_, LD_, PD_)                                                                                   \
    dec8_kernel<NT, W, STEPS, F_, XR_, LD_, PD_><<<grid, W * 32, 0, stream>>>(                                     \
        h(X0), h(X1), w(T0), w(T1), items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(),       \
        Z.data_ptr<float>(), (int)K, (int)N, (int)P, (int)SK, (int)E, (int)slots, h(sv0), h(sv1), h(su2),         \
        out.data_ptr(), (float)limit, done.data_ptr<int>())
#define DEC(F_, XR_)                                                                                              \
    if (ld == 0) DEC1(F_, XR_, 0, 1);                                                                             \
    else if (ld == 1) DEC1(F_, XR_, 1, 1);                                                                        \
    else if (ld == 2) DEC1(F_, XR_, 1, 2);                                                                        \
    else DEC1(F_, XR_, 1, PD4)
    if constexpr (STEPS == 16) {                     // every matrix: gate/up (fuse 0 / 2, xrow) and two ranks' down
        if (fuse == 0) { if (xrow) { DEC(0, true); } else { DEC(0, false); } }
        else if (fuse == 1) { if (xrow) { DEC(1, true); } else { DEC(1, false); } }
        else { if (xrow) { DEC(2, true); } else { DEC(2, false); } }
    } else {                                         // only the down projection of 3 or 4 ranks (one matrix)
        TORCH_CHECK(mats == 1 && !xrow && fuse != 2, "decode expert kernel: ", STEPS,
                    " k steps a warp only for the down projection");
        if (fuse == 0) { DEC(0, false); } else { DEC(1, false); }
    }
#undef DEC
#undef DEC1
}

void exlr8_dec8_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& T0, const at::Tensor& T1,
                   const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& Z,
                   int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t max_items, int64_t E,
                   int64_t slots, int64_t fuse, bool xrow, const at::Tensor& sv0, const at::Tensor& sv1,
                   const at::Tensor& su2, at::Tensor& out, double limit, at::Tensor& done, int64_t ld) {
    constexpr int NT = 8, W = 4;
    TORCH_CHECK(ld >= 0 && ld <= 3, "decode expert kernel: loads 0 (32-bit), 1 / 2 / 3 (16-byte, 1 / 2 / 4 steps ahead)");
    TORCH_CHECK(K % (16 * SK * W) == 0 && N % (16 * NT) == 0, "decode expert kernel: whole k steps a warp");
    TORCH_CHECK(fuse != 1 || (SK == 1 && mats == 1), "the fused down epilogue takes one split of one matrix");
    TORCH_CHECK(fuse != 2 || mats == 2, "the fused gate/up epilogue takes gate and up");
    const int64_t steps = K / (16 * SK * W);
#define ARGS X0, X1, T0, T1, items, counts, members, Z, mats, K, N, P, SK, max_items, E, slots, fuse, xrow, sv0, sv1, \
             su2, out, limit, done, ld
    if (steps == 16) dec_launch<16>(ARGS);
    else if (steps == 12) dec_launch<12>(ARGS);
    else if (steps == 10) dec_launch<10>(ARGS);
    else if (steps == 8) dec_launch<8>(ARGS);
    else TORCH_CHECK(false, "decode expert kernel: 8, 10, 12 or 16 k steps a warp, not ", steps);
#undef ARGS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

