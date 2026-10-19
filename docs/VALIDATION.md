# Packaging validation

## Update — 2026-10-19

- Seven runtime artifacts (including the disabled native atomic extension) were matched by SHA-256 against both running TP ranks.
- All 921 packaged Python source files passed syntax checks. Eight CPU tests passed locally and inside the ARM64 package image: retained defaults, TP payload compatibility, cache-tag separation, source hashes, dated evidence arithmetic, and existing deployment/key/vision contracts.
- Offline build from the already installed `glm-mxprefill:20261018` base passed. A CPU-only container imported the captured native library and its `prompt_atomic` entry point with CUDA uninitialized. Shell syntax checks passed.
- The first container test invocation could not discover tests because `.dockerignore` deliberately excludes them; mounting the same test directory read-only fixed the validation runner. No runtime change was needed.
- Online container identity/start time/restart state was unchanged. Health remained idle, requests stayed at 10, and the pool remained 1,699,840 tokens. This packaging task generated **zero requests** and performed **zero model reloads**.
- The build was checked before final documentation/hash-manifest refresh; runtime source, scripts, defaults, and tests are the validated versions. No image was pushed by this source-repository update.

**Not validated in this packaging task:** a fresh public-base dependency install, rebuilding the native binary, GPU/model startup from this public recipe, full 1M context, video, new decode/prefill runs, or model capability. Performance numbers are dated results from the already tested running deployment. See [validation-20261019.json](../evidence/validation-20261019.json), [source verification](../evidence/prefill-source-20261019.json), and [prefill results](PREFILL_20261019.md).

## Original package — 2026-10-18

The following is the original validation record. References to the running process and its retired draft banks apply to that date, not the updated 2026-10-19 deployment.


2026-10-18，本次只整理仓库，不重载在线模型、不生成请求、不追加测速。

- 在Spark01基于实际本地镜像进行离线构建；共享库符号、固定依赖及Python源码语法检查通过。
- CPU容器成功导入EXLR8 checkpoint contract，CUDA未初始化，未加载模型；无GPU容器运行通过。
- 从真实K4文件重新生成113,074项逻辑张量元数据。`config.json`与`exlr8-runtime-headers.json`和在线view按JSON内容完全一致；重复执行幂等。
- 347个BF16视觉张量使用同一批源文件，张量合计1,127,254,016字节；完整外部分片2,951,938,888字节。视觉主体已独立保留，不要求重新下载旧NVFP4仓库。
- 5项CPU检查覆盖两rank挂载与key隔离、仅rank0视觉、真实草稿路径、显式seed保留、展示数据计算和旧草稿缺失补丁。
- 最终源码包含在线容器已经应用的“缺失27/28旧草稿则跳过”补丁。该启动路径没有通过重载做GPU验收；31日草稿仍为必需。在线进程原驻留银行不因磁盘清理自动释放。
- 原本地基础镜像有117个文件系统层。初版过多COPY导致构建可完成但创建容器超过层数上限；最终Dockerfile仅增加两层，CPU容器创建已通过。

**未测试范围：**全新公开基础镜像联网安装、从本仓库首次加载双机模型、1M满上下文、视频、整体能力及新性能评测。README性能来自已存历史证据，不能当作新镜像验收结果。共享池1,699,840为当前在线实测；新部署略去旧草稿后容量须以启动health为准，不预报新值。

机器证据见 [validation.json](../evidence/validation.json)，文件校验见 [package-sha256.json](../evidence/package-sha256.json)。CPU检查可用 `python3 -B -m unittest discover -s tests -v` 重跑。
