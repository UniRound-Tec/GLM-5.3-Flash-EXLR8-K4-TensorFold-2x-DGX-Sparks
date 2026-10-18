# Local snapshot changes

Snapshot: 2026-10-18, captured from the running GLM EXLR8 service, not a clean upstream checkout. Files are indexed by SHA-256 in evidence. No model weights, credentials, private host configuration or service logs are included.

The TensorFold overlay includes the EXLR8 K4 reader/family contract; original-source FP8 input and head handling; original K4 dense/MLP paths with lossless BF16 fallback storage; decode-only prefetch filtering; short-context graphs; multi-row head and verify8; expert ordering; DFlash2 historical banks, MSE representations and FNC defaults; shared cache, memory admission and visual integration inherited from the retained recipe.

Experimental RRS, shared-expert fusion and large-prefill MXFP8 code remain in the snapshot but default off. The later routed-expert native-FP8 local experiment is not included. No new numerical optimization was introduced during packaging. The final snapshot also copies the already-applied runtime patch that skips missing retired day-27/day-28 draft files; only the pinned day-31 checkpoint is required. This patch was checked without reloading the running model.

Deployment wrappers, configuration paths and metadata-view preparation are portable replacements for the private host scripts. The generated view retains explicit EXLR8 source provenance even though it uses the EXL3-compatible runtime interface. Vision files remain external read-only dependencies.

Precompiled libraries are preserved from the actual runtime, with corresponding Atlas and native sources. The stale, unreferenced libatlas_prefill.so is omitted; it is not an active dependency of the current source. Native TensorFold/JIT kernels otherwise build from the captured .cu/.cpp source at startup. See docs/BUILD.md.
