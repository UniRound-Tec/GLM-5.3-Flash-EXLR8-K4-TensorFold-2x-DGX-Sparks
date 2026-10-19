# Local snapshot changes

Snapshot: 2026-10-19 (prefill update; original package 2026-10-18), captured from the running GLM EXLR8 service, not a clean upstream checkout. Files are indexed by SHA-256 in evidence. No model weights, credentials, private host configuration or service logs are included.

The TensorFold overlay includes the EXLR8 K4 reader/family contract; original-source FP8 input and head handling; original K4 dense/MLP paths with lossless BF16 fallback storage; decode-only prefetch filtering; short-context graphs; multi-row head and verify8; expert ordering; DFlash2 historical banks, MSE representations and FNC defaults; shared cache, memory admission and visual integration inherited from the retained recipe.

Experimental RRS, shared-expert fusion and large-prefill MXFP8 code remain in the snapshot but default off. The later routed-expert native-FP8 local experiment is not included. No new numerical optimization was introduced during packaging. The final snapshot also copies the already-applied runtime patch that skips missing retired day-27/day-28 draft files; only the pinned day-31 checkpoint is required. This patch was checked without reloading the running model.

Deployment wrappers, configuration paths and metadata-view preparation are portable replacements for the private host scripts. The generated view retains explicit EXLR8 source provenance even though it uses the EXL3-compatible runtime interface. Vision files remain external read-only dependencies.

Precompiled libraries are preserved from the actual runtime, with corresponding Atlas and native sources. The stale, unreferenced libatlas_prefill.so is omitted; it is not an active dependency of the current source. Native TensorFold/JIT kernels otherwise build from the captured .cu/.cpp source at startup. See docs/BUILD.md.

## Prefill update — 2026-10-19

Synced the retained running `qmm.py`, `forward.py`, `trial_control.py`, and `prof.py`, plus the disabled `prefill_atomic.py` and its matching library. The draft startup patch is unchanged. File hashes were checked on both TP ranks before packaging. The defaults now enable the 14-shape BF16 configuration table, 18-shape BF16 cuBLAS whitelist, one HC piece, and whole-row projection schedule. This introduces no new weight/activation quantization; cuBLAS and scheduling can change rounding. Maximum measured local relative L2 difference was 0.0147%, not bitwise identity or a model-quality score.

The exact snapshot includes the atomic-reduction experiment because startup imports its extension; it remains disabled after a negative full-service comparison. Its corresponding sources are `vendor/native/expert-atomic.{cpp,cu}`. Optional event profiling is also off. See `docs/PREFILL_20261019.md` for measured results and limitations. All changes were tested before this packaging task; publishing does not run the experiments again.
