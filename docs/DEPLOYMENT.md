# 部署与迁移

本recipe固定双GB10/SM121、ARM64、CUDA13.3/NVIDIA PyTorch26.07栈，TP2/RoCE。非通用GPU或TP3脚本。

## 配置

`bash quickstart.sh init`创建被Git忽略的`config/local.json`。两台分别填写：

- `model_dir`：原始Kindling K4 snapshot，不能填运行时checkpoint-view。目录须含47个dense文件、各层K4及index、config/tokenizer。
- `draft_dir`：31日草稿snapshot绝对路径，末级名必须为固定revision；旧27/28日snapshot可缺省，已同步在线容器的缺失跳过补丁；若另有保留且放在同父目录仍会预加载。2026-10-19在线加载已跳过被删除的旧银行，实际池仍1699840；本次仓库发布不重载服务。
- `vision_dir`：rank0使用的旧BF16视觉来源checkpoint/view，需包含完整config、processor及safetensors index；可指向保留的视觉view及独立完整分片，无需恢复已删除的NVFP4主体。rank1可空。设置`vision=false`可显式选择纯文本部署，但不是所记录的完整视觉配置。
- `data_dir`：新的专用运行目录。metadata view、编译缓存、日志型ledger、默认配置和私有key均放这里；不要指向根目录或现有服务目录。
- `api_key_file`：head已有key文件，或留空生成新key；内容不进入CLI参数、源码或镜像。`prepare`不会覆盖已生成的私有key。
- `worker`、RoCE IP、接口/HCA、端口：按实际机器填写。脚本只使用现有SSH，不配置密钥和网络。

在head执行`prepare`会通过SSH在两台构建各自的元数据视图，模型原文件只读；不会进行模型下载或权重转换。HF snapshot的`../../blobs`链接通过挂载其整个模型缓存目录保持可解析。视觉分片可能位于另一个HF缓存仓库，脚本记录并只读挂载其实际来源。

`sync-image`只经SSH/LAN传输镜像。原始权重需事先存在两端；必要时自行用`rsync -aL --partial --append-verify`复制完整模型目录和31日草稿snapshot，注意保留revision目录名称。复制前核对空间，不自动重复公网下载。

## 构建

原环境优先：`bash quickstart.sh build --base-image glm-mxprefill:20261018`。构建离线、无GPU、不会重载在线模型。新名字`glm53-exlr8-recipe:20261019`避免覆盖原镜像。

公共基础镜像固定在`sources.lock.json`。若使用全新机器，先准备该镜像及`requirements-runtime.txt`依赖，再运行离线build。也可显式用Dockerfile的`INSTALL_DEPS=1`进行联网构建，例如：

```bash
docker build --build-arg INSTALL_DEPS=1 \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
  --network host -t glm53-exlr8-recipe:20261019 .
```

这一步会下载依赖，和默认离线build不同；本次未执行、未声称全新基础镜像安装验收。Docker daemon拉镜像的代理也需单独配置；build参数不控制daemon拉取。

在原实验环境，任何大文件公网下载前必须确认下载机器选择用户当前授权的精确节点，开始后核对CDN连接实际链路；不可自动切换其它节点。FRP规则保持DIRECT。脚本不替用户修改Mihomo或FRP。

## 启停与回退

`start`先检查两台端口、设备、镜像一致性和生成视图，再启动worker、head。已运行或同名异配置容器会被拒绝，不自动删除/替换。约20分钟观察上限仅控制等待；超时需看现有容器状态，不要盲目重启。

`stop`只停止带本recipe归属标签的容器，保留全部文件、容器和镜像。当前原部署容器、原key、原权重与原镜像不在其管理范围。

回退到原部署时，先停止本recipe，再使用原部署的容器及其配套默认文件。原版本可能不识别新字段，不混用不同版本defaults。仓库不提供自动切回另一模型的行为。

API流式性能指标来自后端`sse tensorfold`事件。API默认max_tokens32768与1M上下文上限是不同限制。共享池1,699,840不是每槽容量；实际可用内存与机器环境有关，启动失败不会悄悄缩小池。
