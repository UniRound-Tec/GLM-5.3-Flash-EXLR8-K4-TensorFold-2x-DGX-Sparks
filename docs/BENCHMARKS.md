# 性能展示口径

## 2026-10-19 更新

当前 README 的 FNC7 冷 prefill 已更新为32K **1894.71**、197K **1854.57 tok/s**，首字17.396/106.449秒；均为已完成实验的单次实测，不是均值。本次发布没有新测速。实现、同加载对照和精度边界见 [dated prefill update](PREFILL_20261019.md)，机器数据见 [prefill-20261019.json](../evidence/prefill-20261019.json)。以下是2026-10-18首版发布保留的历史口径，“当前”字样仅指当时的配置。

## 2026-10-18 历史展示

README按请求选择较快有效样本展示。不是均值，不排除其余有效记录来构造平均，也不是宣称每个任务都能达到峰值。本次打包没有运行benchmark。

`evidence/benchmarks.json`收录10条符合当前保留策略的decode记录，以及6条冷prefill记录；每条保留来源文件SHA256、原row索引、seed、完整引擎统计及用量，decode还保留输出。

筛选decode要求：512输出、FNC7、noise1.0、expert_order2、无RRS、原始测试已完成。Code/Prose/算法分别选119.0587/62.1269/64.5235 tok/s。Code是clamp_00…clamp_49重复代码，43轮产生511个decode tokens，包含copy-draft的收益；不能冒充通用编程题速度。该Code样本命中64个输入token，decode计时排除prefill；Prose样本缓存0。所有提示见`evidence/prompts.json`。

Prose选择seed44的62.1269，其他seed没有删除，保留策略三seed均值约57.82。各次测试镜像时间不同，虽然保留路径相同，也不是本仓库全新镜像的联合验收。

冷prefill分三组：

| 记录组 | 8K tok/s | 32K tok/s | 说明 |
|---|---:|---:|---|
| 历史source-mlp/K0 |2302.03|2480.47|较快引擎分段；旧镜像，kda=true、noise0.7等旧参数记录保留|
| 保留FNC7路径 |1636.74|1735.93|较快的当前路径历史记录；长prefill BF16|
| 当前实际镜像mx_prefill=false |1620.68|1727.67|同镜像内基线，当前默认选项|

K0历史首字5.0053/18.8624秒，FNC7保留路径5.1161/18.9756秒。引擎prefill区间不同，不按2.48K/1.74K推断用户首字体验的比值。冷输入缓存全部0；每档仅一次，未计算置信区间。

没有选取已经关闭的`mx_prefill=true`候选作为当前性能，也没有把旧额外Q4 MSE目标版或NVFP4版的成绩混入表中。旧HumanEval/EvalPlus结果不能作为当前K4的能力成绩。

复测脚本默认当前FNC7，可显式`--policy 0`关闭投机，但仍是当前镜像，不能重建历史source-mlp的计时口径。新nonce隔离长输入前缀；如果缓存命中或发现并发，结果保留但`valid=false`，不输出冷prefill吞吐。
