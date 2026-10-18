# GLM-5.3-Flash EXLR8 K4 · TensorFold · 2× DGX Spark

**双 Spark，119 tok/s Code、62 tok/s Prose；历史冷 prefill 最高约2.48K tok/s。**

当前 EXLR8 K4 部署的源码和配置快照：双 GB10、TP2/RoCE、DFlash2、OpenAI兼容API、图片输入、FP8 KV共享池。主模型保留原 K4 / MXFP8 / FP8 格式，**没有额外 Q4 MSE 或 head 重量化**。草稿使用MSE Q4。

> 上述为已有记录中选取的较快有效样本，不是平均值或同一镜像的一轮统一测试。Code为重复代码场景；2.48K prefill为历史K0配置。当前默认及样本来源见下表。本次整理没有追加测速。

## 性能速览

2× DGX Spark / GB10，128GB统一内存/台，ConnectX-7 RoCE，单请求。

| 场景 | 最佳展示样本 | 模式 / 说明 |
|---|---:|---|
| **Code decode** | **119.06 tok/s** | FNC7，512输出，seed42；重复Python函数，copy-draft有效 |
| **Prose decode** | **62.13 tok/s** | FNC7，512输出，seed44；英文技术说明 |
| 算法代码 decode |64.52 tok/s|FNC7，512输出，seed42；区间扫描算法，供对照|
| **冷 prefill · 8K** | **2,302 tok/s** | 历史source-mlp / K0，8,256输入、缓存0；首字5.01秒 |
| **冷 prefill · 32K** | **2,480 tok/s** | 历史source-mlp / K0，32,830输入、缓存0；首字18.86秒 |

Code / Prose 来自保留的 `verify8 + expert_order=2 + draft_noise=1.0` 路径；是之前同路径镜像的实测，**不是重新构建本仓库镜像后的验收数字**。重复代码不能代表任意编码任务；同配置Prose三seed均值约57.82 tok/s。

当前默认 **FNC7、长prefill走BF16** 的已有较快样本为 **8K 1,637 / 32K 1,736 tok/s**，首字 **5.12 / 18.98秒**。K0历史记录与FNC7的引擎计时分段不同，首字反而接近；不能把2,480对1,736解释为当前配置整体慢约30%。当前实际镜像内`mx_prefill=false`配对基线为1,621/1,728 tok/s。

精确提示词、seed、全体纳入展示筛选的样本及计算公式见 [BENCHMARKS](docs/BENCHMARKS.md) 和 [机器可读证据](evidence/benchmarks.json)。温度1、top_p 0.95、top_k 20、thinking关闭；decode用`(输出token−1)/引擎decode秒`，prefill用`未缓存输入token/引擎prefill秒`，首字单列。

## 当前保留的路径

- EXLR8 K4路由专家沿用E4M3码本及EXL3兼容FP16 MMA；**不是完整gr8原生专家引擎**。
- 源MXFP8输入投影≤32行使用原生块缩放MMA；长prefill恢复BF16展开计算。新增长输入MXFP8候选关闭。
- 原K4大投影小窗直读，大窗使用无损BF16备用；decode预取过滤掉prefill备用表示。
- `verify8`覆盖FNC7的2–8行验证，短上下文图减少多余索引工作，保留长上下文回退；专家优先换序。
- 默认`fnc7:0.3`、β2、噪声1.0；31日官方DFlash2 MSE草稿、copy-draft开启。在线进程仍驻留旧草稿银行约1.265GiB/台；磁盘已清理27/28日文件，仓库同步缺失则跳过的启动修复，仅要求31日草稿。未为释放银行重新加载。
- 实际共享池 **1,699,840 tokens**，4槽，单槽配置上限1,048,576；32份/3GiB前缀快照。1M为配置上限，未做满长压力验收。
- rank0图片输入：每图2048 tokens，最多50图。视觉塔复用独立保留的原NVFP4 BF16组件分片；视频尚未验收。

## 快速部署

在第一台Spark执行。需要两台已配置Docker/NVIDIA Container Toolkit、可用RoCE、免密SSH。脚本不修改网络、不删除权重、不停止其他部署。

```bash
bash quickstart.sh init
# 编辑 config/local.json：两端原始K4目录、31日草稿snapshot、rank0视觉来源、网络接口。
# 若沿用现有API key，填写 api_key_file；否则prepare会生成私有文件，控制台不打印key。

# 在原实验环境复用现有镜像；整个构建离线进行：
bash quickstart.sh build --base-image glm-mxprefill:20261018
bash quickstart.sh prepare
bash quickstart.sh sync-image
bash quickstart.sh start
bash quickstart.sh status
```

首次启动约需数分钟，并可能编译CUDA扩展。已有服务占用端口时`start`会停止操作，**不会自动停止旧服务**。先自行安排旧服务停机，或改用新端口/容器名。失败容器保留供排查；`bash quickstart.sh stop`仅停止本recipe创建的容器。

默认API `http://<head>:8888/v1`，模型ID `GLM-5.3-Flash-EXLR8-K4`；key文件位于head数据目录`private/api-key`。后台仅监听`127.0.0.1:8889`。

### 权重与基础镜像

仓库不含模型权重。`prepare`只生成元数据视图和软链接；两台各自的原始文件须已完整存在，可通过LAN复制或`rsync --partial --append-verify`断点同步。

| 组件 | 固定来源 |
|---|---|
| K4目标 | [KindlingAI模型卡](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4)，revision `3117e3b85744ca91c49fcbe95b46c33f31091829`；dense＋K4专家及index，含MTP元数据 |
| 当前草稿 | [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)，revision `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| 图片组件 | 已有的 [LIL NVFP4 CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD) checkpoint/view，347个BF16视觉张量；只读复用，不再量化 |

`model_dir`填写原始Kindling snapshot；`draft_dir`必须指向以31日revision命名的snapshot目录。无需下载旧27/28日草稿。`vision_dir`填写具有config、processor、safetensors index及视觉分片的旧checkpoint/view。详见 [部署与迁移](docs/DEPLOYMENT.md)。

Dockerfile还固定了公开Mia基础镜像digest。它必须已在本地，且具有`requirements-runtime.txt`所列依赖；否则离线构建会明确失败。新机器可在合适下载线路下预先拉取基础镜像并安装依赖，见部署文档；**全新公开基础镜像联网安装和双机冷启动尚未重新验收**。

本实验环境大文件下载须使用用户当前授权的精确节点并核对CDN实际链路，FRP保持DIRECT；仓库脚本不改变代理、不自动下载模型或镜像。构建使用固定运行库快照，附对应CUDA源和重建脚本。

## 手动复测

```bash
python3 scripts/benchmark.py --case code --seed 42 --key-file ~/.local/share/glm53-exlr8-recipe/private/api-key --out code.json
python3 scripts/benchmark.py --case prose --seed 44 --key-file ~/.local/share/glm53-exlr8-recipe/private/api-key --out prose.json
python3 scripts/benchmark.py --case prefill32k --seed 42 --key-file ~/.local/share/glm53-exlr8-recipe/private/api-key --out prefill.json
```

这些命令会实际生成，部署步骤不会自动运行它们。每次保存完整用量、引擎计时、缓存与文本；冷prefill若命中缓存则不计算冷吞吐。脚本默认沿用当前FNC7，不冒充历史K0镜像复现。

## 精度、许可与上游

**模型性能与能力仍有待进一步测试。** 保留源权重不等于全BF16：原始K4/MXFP8/FP8、已有小窗激活量化、内核求和舍入、FP8 KV仍然存在。本仓库没有新能力评测，不沿用旧NVFP4的HumanEval成绩作为当前K4成绩。

源码基础：[TensorFold](https://github.com/ashhart/TensorFold)、[Mia双Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)、[Kindling gr8](https://github.com/kindlingai/gr8)、[Atlas](https://github.com/Enntity/atlas)、[B12X](https://github.com/local-inference-lab/b12x)。固定版本见 [sources.lock.json](sources.lock.json)。

组合代码按 **AGPL-3.0-only** 提供，Apache-2.0/MIT上游文件保留原许可；K4固定权重附MIT许可，DFlash2为CC BY-NC-ND 4.0，视觉来源保留LIL License 1.0约束，均不被本仓库代码许可覆盖。详见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

视觉来源归属：This model is based on [GLM-5.3-Flash-NVFP4-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-CSF-QAD. GLM-5.3-Flash-NVFP4-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0. 本recipe仅复用其BF16视觉组件，主语言模型为上述Kindling K4。

[验证范围](docs/VALIDATION.md) · [源码变更](vendor/MODIFICATIONS.md) · [性能证据](docs/BENCHMARKS.md)
