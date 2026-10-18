// Packaging notice (2026-10-15): modified deployment snapshot; see vendor/MODIFICATIONS.md.
// Original copyright/license notices below remain applicable.
// SPDX-License-Identifier: AGPL-3.0-only
template<int MI,bool GU>
void launch_stream(unsigned mode,dim3 grid,cudaStream_t s,PQ2_ARGS,const unsigned* work,PqwGateUp q) {
    bool zero=!GU && (mode&2),pf=mode&(GU ? 4 : 8);
    #define CALL(Z,P) qad_stream<GU,Z,MI,P><<<grid,256,0,s>>>(A_packed,A_scale,B_packed_ptrs,\
        B_scale_ptrs,scale2_vals,C,expert_offsets,sorted_token_ids,num_experts,N,K,work,q)
    if(zero){if(pf){CALL(true,true);}else{CALL(true,false);}}
    else{if(pf){CALL(false,true);}else{CALL(false,false);}}
    #undef CALL
}
extern "C" int atlas_run(const uint64_t* p,int rows,uint64_t stream_bits,int mode) {
    if(mode==64)return atlas_run_persistent(p,rows,stream_bits);
    if(rows<1||rows>8192||((mode&1)&&rows>32))return -1;
    if(!mode)return atlas_run_control(p,rows,stream_bits);
    auto s=(cudaStream_t)stream_bits;int r=rows*TOP;
    auto x=(const __nv_bfloat16*)p[0];auto ids=(const int*)p[1];auto probs=(const float*)p[2];auto out=(__nv_bfloat16*)p[3];
    auto w13=(const unsigned long long*)p[4];auto sf13=(const unsigned long long*)p[5];auto a13=(const float*)p[6];
    auto w2=(const unsigned long long*)p[7];auto sf2=(const unsigned long long*)p[8];auto a2=(const float*)p[9];
    auto gs1=(const float*)p[10];auto gs2=(const float*)p[11];
    auto counts=(int*)p[12];auto offsets=(int*)p[13];auto cursor=(int*)p[14];auto pairs=(int*)p[15];
    auto inverse=(int*)p[16];auto experts=(int*)p[17];auto prefix=(int*)p[18];
    auto ap=(unsigned char*)p[19];auto as=(unsigned char*)p[20];auto mp=(unsigned char*)p[21];auto ms=(unsigned char*)p[22];
    auto down=(__nv_bfloat16*)p[23];
    cudaError_t error=cudaMemsetAsync(counts,0,E*sizeof(int),s);if(error!=cudaSuccess)return error;
    count_routes<<<(r+255)/256,256,0,s>>>(ids,counts,r);
    scan_routes<<<1,512,0,s>>>(counts,offsets,cursor,prefix,mode&1);
    scatter_routes<<<(r+255)/256,256,0,s>>>(ids,cursor,pairs,inverse,experts,r);
    quant_routes<<<r,256,0,s>>>(x,pairs,experts,gs1,ap,as,r);
    if(mode&1) {
        PqwGateUp q={w13,sf13,a13,mp,ms,gs2};
        #define RUN(MI) \
        launch_stream<MI,true>(mode,dim3(I/128,r<E?r:E),s,ap,as,w13,sf13,a13,nullptr,offsets,nullptr,E,I,H,(unsigned*)prefix,q);\
        launch_stream<MI,false>(mode,dim3(H/256,r<E?r:E),s,mp,ms,w2,sf2,a2,down,offsets,nullptr,E,H,I,(unsigned*)prefix,PqwGateUp{})
        if(rows<=16){RUN(1);}else{RUN(2);}
        #undef RUN
    } else {
        int bound=(r+63)/64+E;
        moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w<<<dim3(I/128,bound),256,0,s>>>(
            ap,as,w13,sf13,a13,nullptr,offsets,nullptr,E,I,H,prefix,w13,sf13,a13,mp,ms,gs2);
        moe_w4a4_grouped_gemm_prequant_t_k128w_compact<<<dim3(H/256,bound),256,0,s>>>(
            mp,ms,w2,sf2,a2,down,offsets,nullptr,E,H,I,prefix);
    }
    if(mode&16)reduce_vec8<<<(rows*H/8+255)/256,256,0,s>>>(down,inverse,probs,out,rows);
    else reduce_routes<<<(rows*H+255)/256,256,0,s>>>(down,inverse,probs,out,rows);
    return cudaGetLastError();
}
