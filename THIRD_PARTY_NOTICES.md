# Third-party notices

The combined recipe and Atlas-derived integration are distributed under AGPL-3.0-only. Original Apache-2.0 and MIT notices remain applicable to their individual upstream files; no claim of sole authorship is made. Model weights are not included or relicensed.

| Component | Upstream | Terms |
|---|---|---|
| TensorFold and EXLR8 codec/layout | [TensorFold v0.6.0](https://github.com/ashhart/TensorFold/tree/v0.6.0), local snapshot | Apache-2.0 upstream |
| GLM multi-GPU integration | [Mia recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold), selected v1.10 changes | Apache-2.0 upstream |
| Native-format references | [KindlingAI gr8](https://github.com/kindlingai/gr8) | MIT; [original notice](licenses/gr8-MIT.txt) |
| Atlas expert / sparse MLA kernels | [Atlas 72588541](https://github.com/Enntity/atlas/tree/72588541a466f53cb9d38afd74c515e14ab601db) and local adaptations | AGPL-3.0-only |
| B12X, CSF loader | [B12X](https://github.com/local-inference-lab/b12x), [LIL vLLM](https://github.com/local-inference-lab/vllm) at refs in sources.lock.json | Apache-2.0 |
| K4 target weights | [Kindling Flash EXLR8](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4/tree/3117e3b85744ca91c49fcbe95b46c33f31091829) | MIT file verified in the actual pinned snapshot; [copy](licenses/EXLR8-model-MIT.txt) |
| DFlash2 weights | [IncoAI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at the required day-31 revision (two retired revisions remain optional) | CC BY-NC-ND 4.0; independent noncommercial/no-derivatives terms, not a blanket commercial-use grant |
| Optional BF16 visual tower | [LIL CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD) at the fixed revision | LIL License 1.0; preserve attribution and source restrictions, no weights redistributed |

This model is based on [GLM-5.3-Flash-NVFP4-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD. GLM-5.3-Flash-NVFP4-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0. Only its retained BF16 visual components are used by this EXLR8 recipe.

NVIDIA PyTorch/CUDA/NGC container components retain their own licenses, including the [NGC container license](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf). The recipe does not grant additional rights to them. Dependency versions are recorded, not an exhaustive legal inventory of the inherited image.

Corresponding kernel sources and build instructions are included under vendor. ARM64 compiled libraries are the captured local runtime; source hashes and build inputs are recorded. The unused historical libatlas_prefill.so is omitted. The disabled mx_prefill library remains for exact initialization compatibility; its B12X source, AOT objects/headers, adapter and original export harness are provided. Publication of binary derivatives must comply with the respective source and dependency terms.
