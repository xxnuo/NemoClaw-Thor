# DS4 gpu2/gpu5 并行实测

验证日期：2026-08-11 至 2026-08-13。

本文记录两台 Jetson AGX Thor T5000 上运行 DeepSeek-V4-Flash-0731 IQ2
混合量化 GGUF 的实际结果。测试覆盖单机冷启动、双机流水线并行、双机 NCCL
张量并行、并发 admission、MAXN 锁频补测、实验性 TP DSpark 单流 smoke，
以及正式 v0.5.6.2 单机 DSpark A/B/并发上限。
prefill/decode 吞吐来自本次容器日志，HTTP wall time 和并发 aggregate
throughput 来自仓库现有 benchmark 脚本；没有把历史 gpu2/gpu3 的
128+64 GiB 数据混入本次 gpu2/gpu5 结论。

## 结论

- 峰值结论按测量含义拆开，不能合并为一个“最高并发”数字：

  | 路径 | 最高吞吐观测 | 最高无内存处置告警完成并发 | 更高边界 |
  |---|---|---|---|
  | 两个完整 Pipeline 副本 | `1 KiB / 512, C=2` 三波中位 **19.6329 tok/s** | **C=2**，每副本一请求 | 第三副本仅静态预算超限，未实测 |
  | 双机 TP fixed batch | `1 KiB / 128, C=2` 三波中位 **4.8279 tok/s** | **C=14**，14/14 完整且实际 batch=14 | C=15 完成但触发 gpu5 earlyoom；resident=16 创建失败 |
  | 单机 DS4 服务，短输出最高 aggregate | DSpark OFF，`1 KiB / 128, C=5` 三波中位 **24.6291 tok/s** | **C=5**，15/15 完整 | C=6 首波 `cudaMallocAsync` OOM |
  | 单机 DS4 服务，长输出最高 aggregate | DSpark OFF，`8 KiB / 512, C=4` 三波中位 **22.6633 tok/s** | **C=4**，12/12 完整 | C=5 首波只完成 4/5，`cudaMallocAsync` OOM |
  | 单机真实 simultaneous DSpark | `1 KiB / 128, C=2` 三波中位 **20.8648 tok/s**；`8 KiB / 512, C=2` **20.4882 tok/s** | **C=2**，`banks_live=2`，draft/hit 持续增长 | C=3 首波只完成 1/3，`cudaMallocAsync` OOM |

- 在本轮 `ctx=131072` 单请求 8 KiB 短请求矩阵中，`gpu2 0:20 / gpu5 21:output`
  流水线的三次 wall time 均值最低，且两组各 3 次的波动最小：`8 KiB / 128`
  约 26.1 秒，`8 KiB / 512` 约 58.0 秒。
- NCCL TP 在两台对称 T5000 上达到 ready 并完成 8 KiB 短请求矩阵；decode 约
  `5.4-5.5 tok/s`，balanced pipeline 约 `12.03-12.08 tok/s`。两条路径使用
  不同 DS4 commit，不是只改变并行方式的单变量对照。
- MAXN + `jetson_clocks` 下，两个独立完整 Pipeline 副本的 C=2 scored 重跑中，
  `1 KiB / 128` 三波 aggregate 为 `9.2444/13.9933/13.9500 tok/s`，中位
  **`13.9500 tok/s`**；`1 KiB / 512` 为
  `19.6329/11.3508/20.0189 tok/s`，中位 **`19.6329 tok/s`**。两组波间差异
  都很大，不能解释为固定的冷启动/稳态阶段，也不能称为稳定峰值。每个 endpoint
  每波只接收一个请求，因此这是两个完整副本间的真实并行，不是单进程 resident
  batching。
- current-head `e3cd1c...` 加 benchmark-only decode-batch glue 的 fixed TP 扫描使用
  base GGUF、target-only、`ctx=4096` 和固定 `1 KiB / 128` workload。resident=12
  的 C=1/2/3/4/5/6/8/12 均完成，C=2 三波中位最高为 **`4.8279 tok/s`**，
  是本组最高重复吞吐；但各波仍有明显方差，不应外推为长期 SLA。resident=14/C=14
  以 `3.3079 tok/s` 完成 14/14，服务端
  确认实际 `decode batch count=14`，且没有 earlyoom/SIGTERM，是本轮最高无内存
  处置告警完成并发。resident=15/C=15 也以 `2.4236 tok/s` 完成 15/15，但 gpu5
  earlyoom 在该生命周期内反复向 worker 发送 SIGTERM，所以 C=15 只证明最高完成
  并发，不代表可安全长期运行的容量。
- 并发扫描使用不同 context 和并发机制，结果不能做严格横向 A/B：pipeline 两个
  独立副本在 `ctx=4096, 8 KiB / 512, C=2` 达到 `8.6857 tok/s` 中位数；每个副本
  只接收一个请求，因此这是副本间真实并行。历史 patched TP 长输出主结果为
  `ctx=32768, C=4, 3.9592 tok/s`，并确认服务端 `decode batch count=4`；这是该
  配置下的最高已验证实际 batch，不是 current-head fixed sweep 的横向对照。另一个 2 ms coalesce 生命周期的探索性
  `C=1` 中位为 `3.9680 tok/s`。以上 aggregate
  都是整波输出 token 除以整波 wall，包含 prefill、排队和调度，不能直接当作
  server decode tok/s。
- 历史 sweep 在各自配置内未见 aggregate 随 offered concurrency 线性扩展：pipeline
  `ctx=4096` 的 `C=4` 两波中位数为 `4.5367 tok/s`，旧 patched TP
  `ctx=32768` 的 `C=8` 为 `1.7081 tok/s`、最大实际 batch count 为 7。这些旧 ref
  结果不与 current-head fixed workload 拼成一条 scaling 曲线。
- patched TP `ctx=32768` 探索性长输出补测的单流 `C=1` 中位 aggregate 为
  `3.9680 tok/s`，`C=2` 中位为 `2.8470 tok/s`；另一次干净 `C=4` 三波中位为
  `3.9592 tok/s` 并确认实际
  `decode batch count=4`。C=1 才是该长输出配置的最高中位 aggregate；C=4 只称为
  “`ctx=32768` 配置下最高已验证实际 batch=4”。`C=1/C=2` 与 C=4 来自不同服务
  生命周期和 coalesce 配置，不拼成严格并发缩放曲线。
- balanced pipeline 的单个 21K-token 请求可完成，但紧接的第二个长请求未进入
  `prompt start`；本轮不能确认随后出现的 data connection error 与卡住存在因果
  关系，也不能称为长请求连续稳定。
- `ctx=32768` 的 balanced pipeline 在约 2.64K-token prefill 上重复停滞；将
  distributed send timeout 从 60 秒提高到 600 秒仍未得到完成响应。后续逐链路
  iperf3 正常，因此故障边界只能收敛到该次 DS4 data connection，不能归因为
  物理光口或 timeout 参数。
- 单机 upstream-main、131072 context 冷启动在约 3 分钟观察窗口内未监听 API，
  进程无 OOM/Xid，但占用约 107 GiB；本轮没有得到可比较的单机 HTTP 数据。
- 旧 ref `d6e64ad...` 临时 binary 的实验性 TP DSpark 非 batched C=1 smoke
  完成 64 token，`proposed=36`、
  `accepted_draft=21`、accept rate `58.33%`、`errors=0`，但 server finish 为
  `36.341 s`，明显慢于同 binary、同 prompt 的 target-only `14.226 s`。两者 greedy
  输出 SHA-256 均为
  `415f6516a256d2548991bfa59388c75768144b4b2c331df034c5b360d2e82bb6`。
  该慢速结果不能外推到 PR #754 当前 head `e3cd1c...`。
  `--batched-session` 路径不调用 speculative driver，因此所有高并发峰值仍是
  target-only，不能称为 DSpark 并发成绩。
- 正式 Entrpi DS4 v0.5.6.2 单机路径在 gpu2、`ctx=524288`、同一模型与 prompt
  corpus 下完成 DSpark OFF/ON A/B。DSpark ON 的 `1 KiB / 128`、`8 KiB / 128`、
  `8 KiB / 512` 三次 wall 中位分别为 `5.855/10.239/25.583 s`，较 OFF 的
  `7.276/11.998/33.454 s` 缩短 `19.5%/14.7%/23.5%`；`8 KiB / 512` decode
  从 `17.9` 提升到 `24.8 tok/s`。所有输出 hash 与 OFF 一致，metrics 记录
  `ds4_spec_drafts_total=1835`、accept ratio `1.0`。
- 同一正式 v0.5.6.2 镜像在 gpu2 MAXN + `jetson_clocks`、`ctx=4096`
  下完成并发边界扫描。生产默认 `DS4_DSPARK_MAX_NLIVE=1` 在 `C=1`
  使用 DSpark，`C>=2` 主体自动转为 target-only；`C=3` 的
  `1 KiB / 128` 和 `8 KiB / 512` 三波中位分别是 **`22.3785`**
  和 **`21.3662 tok/s`**。强制 `DS4_DSPARK_MAX_NLIVE=0` 且
  `DS4_DSPARK_VERIFY_FIT_ROWS=8` 时，`C=2`是最高真实同时 speculative
  并发；`8 KiB / 512` 相同 `C=2` 的 aggregate 从 OFF `18.2933`
  提升到 **`20.4882 tok/s`**，约 +12.0%。

## 环境与产物

| 项目 | gpu2 | gpu5 |
|---|---|---|
| 型号 | T5000，`p3834-0008` | T5000，`p3834-0008` |
| 内存 | 122.9 GiB 可见 UMA | 122.9 GiB 可见 UMA |
| 管理地址 | `192.168.1.122` | `192.168.1.208` |
| 光口地址 | `192.168.100-103.10` | `192.168.100-103.20` |
| 光口 | 4 x 25 Gb/s，MTU 8966 | 4 x 25 Gb/s，MTU 8966 |
| Docker | 29.6.0 | 29.7.1 |

模型目录为 `/home/nvidia/thor-hf-cache/ds4`。模型由 gpu2 通过第一条光口 rsync
到 gpu5，传输 93,691,352,992 bytes，平均 388,050,630 bytes/s。目标文件随后
完成精确校验：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | 86,720,111,488 | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| `DSpark-drafter-Q2K-Q8-0731.gguf` | 6,971,241,504 | `8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686` |
| `DeepSeek-V4-Flash-DSpark-support-0731.gguf` | 5,989,114,272 | `7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360` |

高并发双机路径只使用 base GGUF。`DSpark-drafter-Q2K-Q8-0731.gguf` 是 Entrpi
DS4 v0.5.6.2 单机正式 DSpark 路径使用的 drafter，但不适用于本报告旧 PR #754
external-support TP 实验路径；双机 TP fixed workload 全部为 target-only。另一个
support GGUF 固定于 `antirez/deepseek-v4-gguf` revision
`e7f04037032990db0346398d249baf9fb9df1ccc`，只在 gpu2 的实验性 TP leader
单流 smoke 中加载；gpu5 worker 不需要加载 support model。

| 路径 | 镜像 | DS4 ref | image ID |
|---|---|---|---|
| layer pipeline | `nemoclaw-thor/ds4:main-sm110-dist` | `b0309611041655f4e45671cfd9c9886aff161406` | `sha256:b72249448e072f3f0dff07e53a49e6cb257e1b24126b8c05155abee95fc63fe5` |
| network TP | `nemoclaw-thor/ds4:pr754-sm110-tp` | `d6e64adaa7cd3e16001bc2090e27b76c618a440a` | `sha256:f25d9eae9d93f82218038745ec04f6deb9c1c807e4860a899f1cdf0115a046e6` |
| single-host DSpark | `nemoclaw-thor/ds4:v0.5.6.2-sm110-thor` | `027714a4c290a756ef3e6ca557426528745f2033` | `sha256:4b807e0417aad4fb1f90174200a086a02a1440813df5e55d7c5d9d104d8e08ab` |

两个双机镜像的 image ID 在 gpu2/gpu5 一致，且都必须绕过旧 entrypoint，显式
执行 `/usr/local/bin/ds4-server`。单机 v0.5.6.2 DSpark 镜像只在 gpu2 使用。

## 网络基线

四条链路双向 ping 均为 0% 丢包，RTT 约 0.44-0.48 ms。复用两端已有 iperf3
server，逐链路执行 5 秒、4 stream TCP 测试：

| 链路 | sender | receiver | retransmits |
|---|---:|---:|---:|
| `192.168.100.10 -> .20` | 17.33 Gb/s | 17.31 Gb/s | 63 |
| `192.168.101.10 -> .20` | 17.81 Gb/s | 17.79 Gb/s | 33 |
| `192.168.102.10 -> .20` | 17.40 Gb/s | 17.37 Gb/s | 51 |
| `192.168.103.10 -> .20` | 17.45 Gb/s | 17.42 Gb/s | 39 |

这些是逐链路独立结果。现有 iperf3 server 一次只接受一个测试，四路同时发起时
其余请求返回 `the server is busy running a test`；不能把逐项相加的约 70 Gb/s
解释为 DS4/NCCL 的实际聚合带宽。

## Benchmark 方法

各路径均以下列矩阵、每个 case 3 次为目标；失败或中止的 case 按实际完成次数
记录，表格状态同时注明实际完成和尝试情况。实际运行命令为：

```bash
# balanced pipeline
uv run python serving/benchmarks/bench-ds4-http.py \
  --base-url http://127.0.0.1:18350/v1 \
  --label gpu2-gpu5-pipeline-balanced-20260812 \
  --case 8:128 \
  --case 8:512 \
  --case 72:128 \
  --repeats 3 \
  --timeout 1800

# NCCL TP
uv run python serving/benchmarks/bench-ds4-http.py \
  --base-url http://127.0.0.1:18250/v1 \
  --label gpu2-gpu5-tp-pr754-20260811 \
  --case 8:128 \
  --case 8:512 \
  --case 72:128 \
  --repeats 3 \
  --timeout 3600
```

Prompt 分别约为 2.4K、2.4K 和 21K tokens。binary 的 HTTP response 没有
`timings` 字段，因此 prefill/decode 从 coordinator 的 `prompt done` 和
`decoding ... avg=` 日志提取。wall time 以 HTTP 客户端为准，日志中的
`finish=length` 通常比客户端少约 0.3-0.4 秒。

并发 aggregate 的定义固定为：

```text
sum(completion_tokens in one wave) / whole-wave wall_seconds
```

它包含 prefill、排队、调度、HTTP/JSON 和少量客户端线程池开销，不等于服务端
日志里的 decode tok/s；本报告中的 synthetic integer sequence 也不代表真实聊天
吞吐。pipeline 的 `C` 是分配到一个或多个独立副本的 offered concurrency；只有
每个 endpoint 各承载一个请求时，才能把请求数直接视为副本间同时执行。单 endpoint
收到多个请求时可能排队。TP 的 `C` 是同一服务的请求并发，只有服务端日志确认的
`decode batch count` 才能证明实际合批。
当前脚本只为整波全部成功的 wave 计算有效 aggregate；存在失败请求的 wave 保留
错误与已完成 token 用于诊断，但 aggregate 记为 `null`。任一 wave
失败时整组 `median_aggregate_output_tok_s` 也记为 `null`，避免把失败前的
单波瞬时值冒充稳定中位数。
历史原始并发日志由修改前脚本生成，其中 `aggregate_decode` 字段名沿用了旧命名，
但数值语义同样是上述端到端 aggregate output；当前脚本复跑时输出
`aggregate_output_tok_s`，失败 wave 返回 JSON、退出码 1，整组 median 置空。

并发探针使用：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:18350/v1 \
  --label gpu2-gpu5-pipeline-concurrency-20260811 \
  --concurrency 2 --waves 1 --prompt-kib 8 --output 32 --timeout 1800
```

## 流水线并行

### 对称切分：`0:20 / 21:output`

这是两台同型 T5000 的主要结果，运行 context 为 `ctx=131072`。coordinator
计划内存 43.24 GiB，worker 计划内存 44.12 GiB；route 为：

```text
local 0:20 -> 192.168.100.20:8361 Q2 21:output
```

| Prompt / Output | HTTP wall（完成次数见状态） | Prefill | Decode | 状态 |
|---|---|---:|---:|---|
| 8 KiB / 128 | 26.08, 26.11, 26.09 s | 159.6-159.9 tok/s | 12.06-12.08 tok/s | 3/3 成功，hash 一致 |
| 8 KiB / 512 | 58.03, 58.03, 57.97 s | 160.0-160.5 tok/s | 12.03-12.05 tok/s | 3/3 成功，hash 一致 |
| 72 KiB / 128 | 82.55 s | 301.6 tok/s | 10.36 tok/s | 1 次完成；第 2 次未完成 |

第二个 72 KiB 请求在客户端等待超过 11 分钟，coordinator 只记录
`live kv cache miss`，没有进入新的 `prompt start`。在本次卡住和人工中止过程
中，worker 另记录：

```text
distributed worker: data connection ... closed after error
```

两容器当时仍为 running，且无 OOM/Xid。本轮没有确定该日志与请求未进入执行
阶段的因果关系。客户端等待被人工中止，随后精确删除该组测试容器。因此本配置
的短请求矩阵结果可用；长请求仅完成 1 次，后续 1 次未完成，不能据此宣称连续
稳定。

### 旧异构切分：`0:28 / 29:output`

此切分来自历史 T5000+T4000 配置，本轮在两台 T5000 上作为对照运行。
coordinator/worker 计划内存分别为 57.99/29.37 GiB。

| Prompt / Output | HTTP wall，原始 3 次 | 稳态 decode | 说明 |
|---|---|---:|---|
| 8 KiB / 128 | 132.13, 25.52, 25.59 s | 12.25-12.28 tok/s | 首次 prefill 121.36 s；其余约14.8 s |
| 8 KiB / 512 | 63.41, 56.97, 164.08 s | 约12.2 tok/s | 第三次 prefill 121.90 s |
| 72 KiB / 128 | 89.98, 105.19, 140.42 s | 10.53-10.56 tok/s | prefill 77.49-127.22 s |

输出均完成且 hash 一致，但 prefill 方差很大，不能只报 median。双请求 probe
为 38.06 s、64 output tokens、aggregate 1.68 tok/s；日志显示两个请求依次
执行，不是并行 resident sessions。

对称切分与旧切分的短请求 decode 接近，但对称切分的内存更均衡，8 KiB 短请求
wall time 更稳定，故推荐 `0:20 / 21:output` 作为后续短请求测试起点。

## NCCL 张量并行

本轮 TP 基线镜像来自上游开放 PR
[`antirez/ds4#754`](https://github.com/antirez/ds4/pull/754) 的旧 ref
`d6e64adaa7cd3e16001bc2090e27b76c618a440a`。截至 2026-08-13，该 PR 当前 head
已推进到 `e3cd1c960e8a7d4c768a7c9e16cc86a247e4c2c2`，仍未合入 `main`，GitHub
报告 `mergeable_state=clean`。current-head 的独立 smoke binary
`/tmp/ds4-pr754-head-e3cd1c9-server` SHA-256 为
`7626daf1a7d19a355ab548793aec99944ebbc547a048e806333ebe8736be8974`；后续 fixed
并发扫描使用 current-head 加 benchmark-only decode-batch glue 的
`/tmp/ds4-pr754-e3c-batch-final-server`，SHA-256 为
`03591e97674aa2a6b7780395dcd7412dd03ed32b476aec7e677e19f118c52774`。
它不是 stock PR binary，也没有接入仓库镜像构建。上游 `main` 已有两机
Metal TP 和单机多 GPU CUDA TP，但没有该 PR 提供的跨节点 CUDA NCCL、world 2/4
network TP 路径。

本轮两个 rank 均使用以下配置。以下表格是 PR #754 基础 binary 的短请求基线；
后文“TP：单服务 batch admission”和“TP 短输出 sweep”使用旧 ref 的
`/tmp/ds4-pr754-maxc-server`，而“TP current-head fixed workload 探测”使用
SHA-256 为 `03591e...` 的另一份 binary。三组 decode 数字不能视为同一 binary
的连续复测。

```text
ctx=131072
DS4_CUDA_DIRECT_MODEL=1
DS4_CUDA_MOE_GRAPHS=0
DS4_CUDA_VMM_ARENA=0
DS4_CUDA_LAYER_GRAPHS=0
DS4_CONT_CAPTURE=0
NCCL_SOCKET_IFNAME=mgbe
NCCL_IB_DISABLE=1
transport=nccl
```

两个 rank 各映射 44.48 GiB model shard，并建立 6.15 GiB aligned dense
artifacts，计划内存均为 53.66 GiB。成功 marker：

```text
rank 0/2 ready, transport=nccl
rank 1/2 ready, transport=nccl
NCCL collective ready: rank n/2 device=0 version=2.28.3
```

| Prompt / Output | HTTP wall（完成次数见状态） | Prefill | Decode | 状态 |
|---|---|---:|---:|---|
| 8 KiB / 128 | 43.84, 43.95, 43.77 s | 119.4-121.1 tok/s | 5.40-5.50 tok/s | 3/3 成功，hash 一致 |
| 8 KiB / 512 | 113.41, 112.98, 113.30 s | 119.6-120.2 tok/s | 5.49-5.52 tok/s | 3/3 成功，hash 一致 |
| 72 KiB / 128 | 186.22 s | 130.6 tok/s | 5.11 tok/s | 1 次完成；未继续至 3 次 |

TP 双请求 probe：两个约 2.66K-token / 32-output 请求依次完成，单请求 wall
约 28.0/27.5 秒，总 wall 57.04 秒，aggregate 1.12 tok/s。本次探针未观察到
并行 resident sessions。

pipeline 使用 main ref `b030961...`，TP 使用未合入的 PR #754 ref
`d6e64ad...`。下表只比较本轮两条可运行路径的端到端观测值，不是只改变并行
方式的单变量 A/B；差异不能完全归因于 pipeline 或 TP 实现。

| Case | balanced pipeline | NCCL TP | 本轮观测比值 |
|---|---:|---:|---:|
| 8 KiB / 128 wall | 26.1 s | 43.8 s | TP / pipeline = 1.68x |
| 8 KiB / 512 wall | 58.0 s | 113.3 s | TP / pipeline = 1.95x |
| 8 KiB decode | 约12.1 tok/s | 约5.5 tok/s | pipeline / TP = 2.20x |
| 21K decode | 10.36 tok/s | 5.11 tok/s | pipeline / TP = 2.03x |

本轮对称 T5000 的 TP decode 观测值约 5.5 tok/s，高于历史 T5000+T4000 的
3.47-4.14 tok/s；硬件、网络与运行条件不同，不能作为单变量对照。结果与两条
路径的通信模式相容：TP 在每个 token 的多个切分点执行 NCCL collective，
pipeline 只在层边界传 activation。但本轮未做网络 profile，不能把差异进一步
归因为 collective 开销或有效带宽；四条 25G 物理链路也不等于该工作负载自动
获得 100G 聚合吞吐。

## 单机边界

使用 `main-sm110-dist`、同一 base GGUF、`ctx=131072` 在 gpu2 启动单机。
日志完成 78.71 GiB aligned artifacts 和 83.80 GiB memory plan；约 3 分钟内
仍未出现 HTTP listener。运行中主机内存约 107 GiB used、15 GiB available，
进程无 OOM/Xid、CPU 持续工作。达到有限观察窗口后精确删除测试容器。

因此本轮单机结论仅为“131K 冷启动未在观察窗口内 ready”，不能用它计算
单机与双机吞吐比。仓库 v0.5.5/v0.5.6.x 的单机服务路径、DSpark 和更长启动
窗口属于另一套实验，不应与本轮 upstream-main 双机对照混算。

## 追加复测：双副本 admission 与 distributed send timeout

### `ctx=4096` 单副本 smoke

在扩展为双副本前，先以单个 `0:20 / 21:output` pipeline 验证 `ctx=4096` 路径。
coordinator 使用 API/control 端口 `9750/9760`，worker data 端口 `9761`；两端设置
`DS4_DIST_SOCKET_TIMEOUT_SEC=600`，coordinator 另设置
`DS4_DIST_PREFILL_WINDOW=2`，worker 设置 `DS4_DIST_WORKER_PREFETCH_DEPTH=2`。
单个 326-token prompt / 16-output 请求在 `86.53 s` 完成，输出 SHA-256 为
`4ab5eccf0f9ab6f0f1cc299bf78f12a947d082dec601c0522dbcf093673c5ffb`。

本次 smoke 的服务端日志摘录显示 prefill `83.938 s`（`3.88 tok/s`）、decode
`6.23 tok/s`，distributed telemetry 的 `downstream_wait=0.000ms`；没有 route
failure、断链或重连。它只证明该 4K 配置和短 prompt 可完成，不能反推后文
`ctx=32768` 的约 2.64K-token prefill 稳定。客户端原始结果：
`/tmp/pipeline-direct-a-smoke-1k16-20260812.log`，SHA-256
`dd9958104f9061523a5a38bc53c4a069241d098d83abfa21a3c28d7093028449`。

### 双副本 pipeline

为测量多个独立 pipeline 副本的 admission，gpu2/gpu5 同时启动两套相同的
`main-sm110-dist` 服务，均为 `0:20 / 21:output`、`ctx=4096`、
`DS4_DIST_SOCKET_TIMEOUT_SEC=600`；coordinator 使用
`DS4_DIST_PREFILL_WINDOW=2`，worker 使用 `DS4_DIST_WORKER_PREFETCH_DEPTH=2`。
API 通过两个独立 SSH tunnel 暴露为 `19850` 和 `19870`；每波两个 8 KiB /
128 请求分别发送到两个副本，执行 2 波。四个请求均返回 128 tokens，输出
SHA-256 均为
`fae23d7317d5ac0bd56e16ab4a357862ac4327d64eaed1bd594d1da69439b10d`。

| Wave | 两请求 wall | aggregate output |
|---:|---:|---:|
| 1 | 52.86, 52.94 s | 4.84 tok/s |
| 2 | 54.44, 54.44 s | 4.70 tok/s |

中位 aggregate 为 `4.77 tok/s`。两套 coordinator/worker 均保持
`restart=0`、`OOMKilled=false`，日志没有 route failure 或 coordinator
disconnect；每个副本的 prefill 约 32.1-32.7 s，decode 约 19.9-21.3 s。
该结果代表两个完整模型副本的并行 admission，不代表单个 pipeline 请求的
decode 速度；每个副本仍各占约 40.20/41.08 GiB resident model。

客户端原始结果：`/tmp/pipeline-stock-r2-c2-8k128-20260812.log`，SHA-256
`370170b80522730f918a11baf28508395863b57a596d46496fdb7c527448897a`。
该历史 Pipeline 结果只以客户端日志为证；`ds4-final-stock-*` 是 TP sweep 日志，
不作为本小节的 Pipeline 服务端证据。

`DS4_DIST_SOCKET_TIMEOUT_SEC` 在本次 DS4 ref 中只设置 `SO_SNDTIMEO`；接收超时
由独立的 `DS4_DIST_SOCKET_RECV_TIMEOUT_SEC` 控制，默认不设置。因此后文所称
“600 秒复测”只是把 distributed send timeout 提高到 600 秒，不是为整条请求或
receive wait 设置 600 秒 deadline。

### `ctx=32768` send timeout=600 复测边界

此前同一 `0:20 / 21:output`、`ctx=32768` 配置在默认 60 秒 send timeout 下，
约 2644-token prompt 于约 128 秒处触发 route failure；coordinator 移除
worker，记录 `replaying 2048 tokens`、`forgot failed route worker` 和
`route incomplete; next needed layer 21`，worker 随后重连。已有的 339-token
请求在重连后成功，但紧接的 2667-token 请求约 128 秒后再次出现同样 route
failure，因此短请求成功不能证明长 prompt 稳定。两个容器当时均为
`restart=0`、`OOMKilled=false`，未见 Xid；日志没有暴露具体 send/read 错误，
不能仅凭 route marker 判定根因。原始日志：

```text
/tmp/ds4-pipe-a-timeout60-coordinator-20260812.log  sha256=4d8f76e58534f59ee6b1b067367b2059821d1cb537ebc3a1dc379b2bd75a14cb
/tmp/ds4-pipe-a-timeout60-worker-20260812.log       sha256=f9ca94695ca61b1ec6f8e1f55492d9c725a86560cae1fd54dcf5e9e73b98231e
```

提高到 600 秒的第一次尝试在 coordinator control 端口 `9860` 和 worker data
端口 `9861` 遇到已有监听，未执行请求；只读确认占用者是当时仍运行的
`ds4-final-stock-a-{coordinator,worker}-20260812`，这是实验编排噪声。随后改用
fresh ports `11050/11060/11061` 重新启动
同一 `0:20 / 21:output` pipeline。route ready 后，9-token smoke 在 1.176 秒内
完成，但约 2.64K-token 请求仍长期没有完成 prefill。现场检查该 DS4 TCP data
flow 时观察到重传、乱序和 `cwnd=1`，两端 CPU 均空闲；延长 timeout 没有证明
问题得到修复。

停滞后四条光口分别进行双向 iperf3，单链路约 `17.1-17.7 Gbit/s`，没有复现
同样的连接停滞。因此本轮只能记录“该 DS4 data connection 停滞”，不能确认
物理链路故障，也不能确认 socket timeout 是根因。该请求没有产生有效 benchmark
JSON；fresh-port 启动及 smoke 日志为：

```text
/tmp/ds4-dspark-pipe-coordinator-20260812.log  sha256=262b8be3c6d9fc96371d75ecd1d26753939e78ceea41619d5b6b736f6f72e063
/tmp/ds4-dspark-pipe-worker-20260812.log       sha256=57e74f94734ad838f709ceb27b598b607ba61cdb499d90faa57e52c122dfadb7
```

这两个日志保留了 ready 和 smoke，但未保留后续卡住请求的完整服务端增量；TCP
状态来自卡住期间的只读现场检查。因此该复测只用于界定失败范围，不纳入吞吐表。

### 并发扫描与最佳已观测吞吐：8 KiB prompt

以下结果按“同一 8 KiB synthetic prompt 规模、目标输出长度相同”比较；prompt
由各次 run 的 label/slot 重新生成，未声称跨拓扑逐字相同。这里的 pipeline
并发扫描使用 `ctx=4096`，patched TP 扫描使用 `ctx=32768`，所以数字只表示各自
配置下的观测，不是同 context 的横向比较。

#### Pipeline：独立副本 admission

两个 `main-sm110-dist` 副本各为 `0:20 / 21:output`、`ctx=4096`。`C=4`
通过两个 API endpoint 轮询发送，每波同时向每个 endpoint 提交两个请求。这是
四请求成功提交的 offered endpoint concurrency；两个副本之间真实并行，但每个
副本内部的两个请求仍依次排队，不代表四个请求同时执行，也不是单 pipeline
服务内 batch。两波全部成功：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:19950/v1 \
  --base-url http://127.0.0.1:19970/v1 \
  --label pipeline-stock-r2-c4-8k128-20260812 \
  --concurrency 4 --waves 2 --prompt-kib 8 --output 128 --timeout 1800
```

| Case | Wave wall | Aggregate output | 状态 |
|---|---:|---:|---|
| 8 KiB / 128, C=4 | 106.78 s, 119.67 s | 4.795, 4.279 tok/s；中位 `4.5367` | 8/8 成功，hash 一致 |

`C=8` 曾按相同 endpoint 轮询尝试，但客户端因 `RemoteDisconnected` 没有产生
有效 benchmark JSON，因此不把它列为成功吞吐。C=4 客户端日志：
`/tmp/pipeline-stock-r2-c4-8k128-20260812.log`，SHA-256
`61ffc48613d3263a1f6d735be3c60fbe68dad1f2b131632f77de5b6e75f00eed`。

长输出 C=2 复测使用相同两个 endpoint，每个副本每波接收一个请求：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:19950/v1 \
  --base-url http://127.0.0.1:19970/v1 \
  --label pipeline-stock-r3-c2-8k512-20260812 \
  --concurrency 2 --waves 3 --prompt-kib 8 --output 512 --timeout 1800
```

| Wave | Wall | 两请求 wall | Total output | Aggregate output |
|---:|---:|---|---:|---:|
| 1 | 117.894 s | 117.814, 117.892 s | 1024 | 8.6857 tok/s |
| 2 | 117.007 s | 116.689, 117.006 s | 1024 | 8.7516 tok/s |
| 3 | 117.920 s | 117.920, 117.842 s | 1024 | 8.6838 tok/s |

6/6 请求均完成 512 token，prompt token 范围 `2644-2690`，输出 hash 均为
`d660d192b7e396a825ee6f6cfd6a7b5995ca93fdb2ebd7a2ab4ea135c4adcca0`；中位
aggregate 为 **`8.6857 tok/s`**。客户端原始日志：
`/tmp/pipeline-stock-r3-c2-8k512-20260812.log`，SHA-256
`82dccfe423f9e8e8043e0c974c94dde984d01b8950d10e754a568465182c1e15`。该复测仅
保留客户端结果，不复用上文 128-token 双副本的服务端日志作为 512-token 证据。

#### TP：单服务 batch admission

使用 `nemoclaw-thor/ds4:pr754-sm110-tp`（base ref
`d6e64adaa7cd3e16001bc2090e27b76c618a440a`）并 bind-mount 本地 patched
binary `/tmp/ds4-pr754-maxc-server`，binary SHA-256 为
`fb600318c4ddccb763e9240606f7079481492fe2902bf37afd8c8805cbaeb16f`。实际使用的
21 行 benchmark 补丁已保存为
[`serving/docker/patches/ds4-pr754-network-tp-batched-decode-benchmark.patch`](../docker/patches/ds4-pr754-network-tp-batched-decode-benchmark.patch)，
SHA-256 为 `29d0b9f608faef030bfda65ce4821f5707701328e7fd0f5c32e3515e3fe83325`；
它只在 `ds4.c` 增加 decode-batch glue，不接入默认镜像构建。运行路径为
`SYNC prefill + EVAL_BATCH decode`，`DS4_CUDA_MIXED_PREFILL_DECODE=0`，不是
fused mixed decode。

C=4 干净复测的配置为 `ctx=32768`、`--batched-session 8`、
`prefill_quantum=2048`、`decode_coalesce_us=10000`、NCCL world 2。rank 0/1
均 ready，API 在 gpu2 `127.0.0.1:10050` 监听，本地 tunnel 为 `20050`；两个
容器均 `restart=0`、`OOMKilled=false`，未发现 Xid、rank disconnect 或
collective failure。

最终长输出复测命令：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:20050/v1 \
  --label tp-patched-r3-c4-8k512-20260812 \
  --concurrency 4 --waves 3 --prompt-kib 8 --output 512 --timeout 3600
```

| Wave | Wall | Total output | Aggregate output |
|---:|---:|---:|---:|
| 1 | 524.06 s | 2048 | 3.9079 tok/s |
| 2 | 517.28 s | 2048 | 3.9592 tok/s |
| 3 | 514.27 s | 2048 | 3.9824 tok/s |

12/12 请求均完成 512 token，prompt token 范围 `2639-2713`，输出 hash 均为
`d660d192b7e396a825ee6f6cfd6a7b5995ca93fdb2ebd7a2ab4ea135c4adcca0`；中位
aggregate 为 **`3.9592 tok/s`**。服务端日志反复出现 `decode batch count=4`，
说明该 `ctx=32768` C=4 确实进入四请求 decode batch；这是该历史配置下最高已验证
实际 batch=4，不是 current-head fixed sweep 的横向对照。客户端 response 没有
`timings`，所以脚本中的 request decode
字段为 0，不用于本结果。

客户端原始日志：`/tmp/tp-patched-r3-c4-8k512-20260812.log`，SHA-256
`96c9c794489c9da3ea2778d5cc9b4c02ffaa448244ef8c49b8718e5039885544`；两端服务
日志 SHA-256 分别为 coordinator
`39b7107e49fb0bb41e699fefd09c2cfd8484a3ee44a0f806f969e0c35d53a2f2`、worker
`61f1b4481faf5c322372e9b0cc17e6b9bb49283340ca73d4a44f1ccf99fcb4ce`。构建使用
PR ref 工作树、CUDA `sm_110` 和容器内 NCCL headers：

```bash
make -C /tmp/ds4-pr754-maxc-src -B ds4-server \
  CUDA_ARCH=sm_110 \
  NCCL_INCLUDE_DIR=/tmp/ds4-pr754-build.x9hodA/nccl
```

成功构建日志为 `/tmp/ds4-pr754-maxc-exact-build.log`，SHA-256 为
`03729fb366a0588ca5463b83607c4ac065658b72ffcefac8114ce83a0d3e62c6`；该日志
以 `make: Leaving directory` 结束，且产物与上面的 binary/source hashes 对应。

另一个 resident=8 服务生命周期的探索性长输出 warm 补测记录如下。原始启动命令没有
设置 `DS4_SERVER_DECODE_COALESCE_US`，因此使用该源码的默认值
`decode_coalesce_us=2000`。其配置为 `ctx=32768`、resident 8、gpu2 API
`12150`、本地 tunnel `22150`；它与上文 C=4 的 `10050/20050, 10000 us`
配置不同，只用于补充观察单流和双流行为：

| C | 各波 aggregate output | 中位数 | 证据 |
|---:|---:|---:|---|
| 1 | 4.0393, 3.9680, 3.8626 tok/s | **3.9680 tok/s** | `/tmp/tp-peak-r8-c1-warm-8k512-20260812.log`，SHA-256 `de1977273d3486bf5134b7194d0f20019e999d63bf5b7357718b74cddcd9d29f` |
| 2 | 2.9907, 2.1567, 2.8470 tok/s | **2.8470 tok/s** | `/tmp/tp-peak-r8-c2-warm-8k512-20260812.log`，SHA-256 `4b9f50a3bb0db4bd95e296780d8baa605cd116a2dbaeae978534dc5293490f5e` |

后续把该服务重启为 `decode_coalesce_us=10000` 后，resident=8/C=4 重跑与另一条
C=1 客户端重叠，客户端被终止且产生 orphan 请求；后来保存的
`ds4-tp-peak` 服务日志属于这个被污染的重启生命周期，只用于诊断，不能为表中
C=1/C=2 背书。`/tmp/tp-peak-r8-c4-{warm,clean}-8k512-20260812.log` 均为空文件，
作废且不计入任何 median 或峰值结论；它们不能替换上文
`/tmp/tp-patched-r3-c4-8k512-20260812.log` 的干净 C=4 三波结果。表中的 C=1/C=2
也不与上文另一生命周期的有效 C=4 合并计算 scaling efficiency。

### TP 短输出 sweep

以下是同一 patched binary 在 `8 KiB / 128` 下的 resident/concurrency 扫描。
各行 resident 和服务生命周期并不完全相同；`aggregate output` 是整波端到端
吞吐，resident 是服务预留的 session 数，`C` 是本次 offered concurrency。
其中 resident=8 的 C=1/2/4 只保留了会话现场转录，是探索性观测，不作为
严格峰值证据。

| Resident | C | 各波 aggregate output | 中位数 | 证据 |
|---:|---:|---:|---:|---|
| 2 | 1 | 1.9724, 2.3669 | 2.1697 | `/tmp/ds4-tp-r2-c1-20260812.log` |
| 2 | 2 | 2.0877, 1.8640 | 1.9759 | `/tmp/ds4-tp-r2-c2-20260812.log` |
| 4 | 4 | 1.8311, 1.6647 | 1.7479 | `/tmp/tp-patched-r4-c4-8k128-20260812.log` |
| 8 | 1 | 1.7945, 1.5011 | 1.6478 | 会话现场转录，未保留独立 JSON |
| 8 | 2 | 1.3940, 2.7512 | 2.0726 | 会话现场转录，未保留独立 JSON |
| 8 | 4 | 2.2574, 2.2359 | **2.2467** | 会话现场转录，未保留独立 JSON |
| 8 | 8 | 1.7081 | 1.7081 | `/tmp/ds4-tp-c8-r8-20260812.log` |

短输出的最佳探索性观测是 `resident=8, C=4` 的 `2.2467 tok/s`，但该值来自
会话现场转录，不列为严格峰值；最大单波已观测 offered concurrency 是 `C=8`，
其可复算 aggregate 为 `1.7081 tok/s`，已低于上述现场观测。resident=8 的 C=1/2/4
数值来自当日会话现场转录，不把它们伪装成仍存在的独立原始 JSON；resident=4
复测客户端日志 SHA-256 为
`4588e459b954aeabdd75f9d9d58606563e1dba92a57192a1fb68f95808b68d2c`。

上述是旧 ref、旧 harness 的历史 sweep：最大单波 offered concurrency 为 `C=8`
（resident=8、`8 KiB / 128`，单波 `1.7081 tok/s`），服务端当时最大实际 decode
batch count 为 7。它不与后文 current-head fixed workload 合并；后者已将最高完成
offered concurrency 推进到 resident=15/C=15。

pipeline C=2/C=4 与 TP C=4 的 `C` 不是等价变量：pipeline C=2 是两个完整模型
副本各执行一个请求；pipeline C=4 是每个副本提交两个请求且副本内部排队；TP C=4
才是一个双机 TP 服务的内部 batch。它们只能作为各自拓扑在本轮工作负载下的
观测，不能作为严格单变量 A/B。

## MAXN 锁频补测

为回答“充分发挥性能”的边界问题，两机切到 `MAXN, mode 0` 并执行
`jetson_clocks`。本节只报告形成完整客户端 JSON 的结果；测试前快照保存在两机
`/tmp/ds4-peak-concurrency-restore-20260813.conf`，其 CPU/GPU/NVD 最低频率分别为
`972 MHz / 315 MHz / 315 MHz`。

### Pipeline 单副本

使用 `main-sm110-dist`、`0:20 / 21:output`、`ctx=4096` 和单个完整 Pipeline
副本，运行 `1 KiB / 512, C=1` 两波：

| Wave | Wall | Aggregate output |
|---:|---:|---:|
| 1 | 42.476 s | 12.0534 tok/s |
| 2 | 42.524 s | 12.0403 tok/s |

中位 aggregate 为 **`12.0469 tok/s`**，2/2 请求均返回 512 token，输出
SHA-256 均为
`d660d192b7e396a825ee6f6cfd6a7b5995ca93fdb2ebd7a2ab4ea135c4adcca0`。
服务端 decode 平均约 `13.01-13.02 tok/s`。客户端日志：
`/tmp/pipeline-cleanmaxn-single-c1-1k512-20260813.log`，SHA-256
`42d4a893ee64aab9d09233fb82e061fe4dd0332ef74afdae8b3670de49c98712`。

### Pipeline 双副本：scored workload 重跑

两个独立完整 Pipeline 副本均为 `0:20 / 21:output`、`ctx=4096`，客户端通过
`28450/28550` 两个 endpoint 轮询；C=2 时每个 endpoint 每波只接收一个请求。
因此这是副本间真实并行，不代表单个 Pipeline 服务内的 resident batch。
该次日志由过渡版 harness 生成，各 slot/wave 使用不同 nonce prompt；请求规模、
输出上限和输出序列相同，并完成 token/hash gate，但不属于逐字相同的 fixed corpus。

| Workload | 三波 aggregate output | 中位数 | 完成情况 |
|---|---|---:|---|
| `1 KiB / 128, C=2` | 9.2444, 13.9933, 13.9500 tok/s | **13.9500 tok/s** | 6/6 完整 token，hash 一致 |
| `1 KiB / 512, C=2` | 19.6329, 11.3508, 20.0189 tok/s | **19.6329 tok/s** | 6/6 完整 token，hash 一致 |

两组结果都存在明显波间方差，不能再解释为“首波突发、后两波稳态”；中位数只代表
本次三波样本，不是稳定容量结论。客户端日志及 SHA-256：

```text
/tmp/pipeline-scored-c2-1k128-20260813.log
  sha256=9bb746ffa765a3884c52bb23e6a0605a1d9f7234a638873618f1e60f5f15f3d3
/tmp/pipeline-scored-c2-1k512-20260813.log
  sha256=f3443902f521ce5e94a35fb1fb372a81b26311207bfe1968059405d9c25e9e60
```

服务端对应日志 SHA-256 为 gpu2 A/B
`9a9e45b226f61b733621463d6aee107d505c1949f849af31fe738c13f24822b8` /
`cb3869ed000da0f4207ad55d358898a6ef7531e31425fd371e3229b37f3f6972`，gpu5 A/B
`e563d42d4506c827aade0993b55ecef893b96a8cbcc5af96855bcb7872ae0bcb` /
`d80dfb195ff1b8a17d62753de492c58f00cdcaee8288aa028110e9868ecfa84e`。
完整日志还包含计分窗口之前的历史请求，因此只对本次计分时间窗作完整性声明。

`ctx=4096` 启动日志给出的单副本内存计划为：gpu2 coordinator resident
`40.20 GiB`、planned `40.62 GiB`；gpu5 worker resident `41.08 GiB`、planned
`41.50 GiB`。按 planned 值静态外推三个完整副本，gpu2 需要
`3 x 40.62 = 121.86 GiB`，几乎占满 `122.9 GiB` 可见 UMA；gpu5 需要
`3 x 41.50 = 124.50 GiB`，已经超过可见 UMA。该预算没有给系统、Docker 或运行时
分配留下可用余量，因此不把三副本列为实际可用配置；但本轮没有启动第三副本，
也没有 `pipe-c` 或明确的第三副本 OOM 记录，不能写成“三副本已实测失败”。初次
A+B 启动时出现的 `exit=137` 与 `NV_ERR_NO_MEMORY` 发生在 TP/DSpark workload
仍驻留时，只能证明当时存在资源争用；清理竞争 workload 后 A+B 已成功运行。

另有双副本驻留、单请求 C=1 的复测中位 `12.0148 tok/s`，用于确认第二副本
本身可稳定服务；它不是 C=2 并发成绩，原始日志为
`/tmp/pipeline-cleanmaxn-dualresident-c1-1k512-20260813.log`，SHA-256
`650004781a5a474dd7a44e2b55bd97941dbd9152d153e21017896106c97d4eed`。

### TP current-head fixed workload 探测

相同锁频条件下，fixed TP 使用 PR #754 current head
`e3cd1c960e8a7d4c768a7c9e16cc86a247e4c2c2` 加 benchmark-only decode-batch
glue；运行 binary SHA-256 为
`03591e97674aa2a6b7780395dcd7412dd03ed32b476aec7e677e19f118c52774`。
该路径只加载 base GGUF，明确为 target-only。resident=12 配置为 `ctx=4096`、
`prefill_quantum=2048`、`mixed_prefill_quantum=512`、
`decode_coalesce_us=2000`；resident=14 和 resident=15 均使用相同 workload 和
调度参数。

fixed harness 为每个 slot 发送完全相同的 `1 KiB` synthetic integer-sequence
prompt，要求 128 completion tokens、`finish_reason=length`、合法连续整数序列和一致
输出 hash。任一请求未通过 gate 时整波 aggregate 记为无效。有效结果如下：

| Resident | C | 各波 aggregate output | 中位数 | 完成情况 |
|---:|---:|---:|---:|---|
| 12 | 1 | 1.1513 | 1.1513 | 1/1，batch=1 |
| 12 | 2 | 3.4463, 4.8279, 4.8301 | **4.8279** | 6/6，batch=2 |
| 12 | 3 | 4.8104 | 4.8104 | 3/3，batch=3 |
| 12 | 4 | 2.7898 | 2.7898 | 4/4，batch=4 |
| 12 | 5 | 4.8173, 2.0931, 1.5761 | **2.0931** | 15/15，batch=5 |
| 12 | 6 | 2.2167 | 2.2167 | 6/6，batch=6 |
| 12 | 8 | 2.0126 | 2.0126 | 8/8，batch=8 |
| 12 | 12 | 1.9993 | 1.9993 | 12/12，batch=12 |
| 14 | 14 | 3.3079 | 3.3079 | 14/14，batch=14；无 earlyoom/SIGTERM，单波 |
| 15 | 15 | 2.4236 | 2.4236 | 15/15，batch=15 |

C=2 的三波中位是本组最高重复吞吐，但 C=5 的重复结果没有复现其首波 `4.8173`，
因此不能声明单调 scaling curve。C=14 是本轮最高无内存处置告警完成并发；其
coordinator 多次确认 `decode batch count=14`，gpu5 日志未见 earlyoom/SIGTERM。
C=15 是本轮最高完成 offered concurrency，
且 coordinator 日志确认实际 `decode batch count=15`。但 gpu5 `lzc-earlyoom` 在
该生命周期内因可用内存低于 5% 多次向 worker 发送 SIGTERM；进程未退出，请求最终
仍全部完成。故 C=15 只作为容量上限观测，不推荐作为长期运行配置。

另有 fixed harness 启用前的 resident=12/C=10 历史单波观测：10/10 请求完成，
aggregate 为 `4.7854 tok/s`。该日志各 slot 的 prompt tokens 为 `333-347`，来自旧
nonce harness，且未记录后续 fixed gate 的完整字段，因此不纳入上表或 fixed
workload 结论。客户端日志 `/tmp/tp-maxn-r12-c10-1k128-20260813.log`，SHA-256
`7ef7e0f4bb7f43d120af3165398e7b73f02a267fb99b62307889afa0de48dbd8`。

客户端日志 SHA-256：

```text
C1   6630c7767524fb447b784e05a7a2f3af30b23d4e35ba53933017862aee2ff859
C2   ed2169cbe38df573820471b0ff34b6dfc053f167d12390ac9397812bbfcbf077
C2x2 7cf60d3a3ba1259efb7c61a538b9c88adaa51b8e93c5f72450704ea29eb84222
C3   a4a8958e60ec24211f2320f38b799962f9a1120646b2eb580f828d4a62e87204
C4   f051a949697c74d6b575c524adccf44c8bf715f56d02b41a82387e40f49bf343
C5   d7164d938c3bc1cff6aa5b1353e2a9e28011e0277246422428626ad1d028f9e5
C5x2 6e484d6066985fb7bd444f08c3afe542c1350309b0fe58761c257173ff63428b
C6   359a3a3e2c05d926f9808bc2c4e32f9e4be7549f87a423d34e31ffcac34efb38
C8   344faabd3f50aff34d6e9d37f8cbd2140e0c7aabc22194337aa845e375355594
C12  991d4b692b0392ec88627c364ed17681bab0eb18125cf50c30ded18ba22d3bbf
C14  410a6c47c454d54254c4ed8a12fa2e501f01727e34e126e293fdf53e2de80aac
C15  40b6f78bd0534f8f48262627974c2933d8ff451124fd809d712379632dbb9067
```

C14 服务端日志 SHA-256 为 coordinator
`7ded894ff015d0c3a00695cd23c250ce5e2571c48b95f02d229224a6a79a20c8`、worker
`58e69bd36c9c7dc80869afa18ee63fc7eb9f43fcba5715705a5ac8b6e62f01bf`。

resident=16 的历史启动边界仍是在第 16 个 session 创建阶段失败：
`tp: worker rank 1 failed during session create` / `failed to create cuda session
16/16`，且 `OOMKilled=false`。C15 的完成将“最高成功创建并运行”边界推进到 15，
但 earlyoom 证据说明它没有足够安全余量。

此前另一个 `ctx=32768/resident=8` 生命周期随后提交 `C=4, 1 KiB / 512`，服务端确认持续出现
`decode batch count=4`；但约 10 分钟后每个请求仅生成约 200/512 token，明显没有
形成更高吞吐。该历史请求被终止，客户端文件为 0 bytes，不进入结果表；它不
影响上文 `ctx=4096` 的 MAXN 结果；C=8/C=12 使用 fixed harness 另行记录，C=10
则是 fixed harness 启用前的历史观测，均不与该 `ctx=32768` 生命周期拼接。

## 实验性 TP DSpark 单流

上游 `main=84cc882` 和 PR #754 都没有正式支持双机 CUDA DSpark。本节使用临时
实验 binary，仅放行非 batched TP leader 的 `--mtp FILE --dspark` 路径：

```text
/tmp/ds4-pr754-dspark-tp-server-20260813
sha256=2225215eedfc0c0f3c5b899df9d65e091ea4298a52b229c5bd82fd98391fcab6
```

这不是上游发布能力，也不接入镜像构建。配置为 `ctx=32768`、NCCL world 2、
非 batched C=1；leader 加载正确 support GGUF，worker 只加载 base GGUF。启动
日志确认：

```text
DSpark support model detected: stages=3 block=5 markov_rank=256
DSpark target-hidden capture enabled: layers=40,41,42
```

请求为固定 29-token prompt、`max_tokens=64`、temperature 0。结果如下：

| 路径 | Finish wall | Server decode tok/s | E2E output tok/s | Output SHA-256 |
|---|---:|---:|---:|---|
| TP DSpark | 36.341 s | 1.83 | 1.7611 | `415f6516a256d2548991bfa59388c75768144b4b2c331df034c5b360d2e82bb6` |
| 同 binary target-only | 14.226 s | 5.47 | 4.4988 | `415f6516a256d2548991bfa59388c75768144b4b2c331df034c5b360d2e82bb6` |

DSpark 进程停止时现场转录的统计为（现有 response/server 哈希日志不包含这些 counters）：

```text
proposed=36 accepted_draft=21 accept_rate=58.33% avg_accept=0.618
errors=0 saved=13535.195ms net_saved=146.390ms
```

这证明旧 ref `d6e64ad...` 上的临时协议在该 deterministic smoke 中保持 greedy
输出一致，且 draft/verify 确实执行；但 `net_saved` 只有 `146.390 ms`，实际 wall
比 target-only 慢 `22.115 s`。因此没有继续跑该临时实现的长 DSpark benchmark；
这个慢速结果只适用于该旧 ref 和临时 binary，不能外推到 PR #754 当前 head
`e3cd1c...`。更重要的是，DS4 的 `--batched-session` 路径不调用 speculative
driver，所以历史 TP C=4/C=8 与 current-head fixed TP 数据都不能标为 DSpark 并发；
Pipeline distributed 路径也仍缺
DSpark 入口。

原始证据：

```text
gpu2:/tmp/ds4-tp-dspark-smoke2-coordinator-20260813.log
  sha256=d8ebc9db31d1e56a6b82a5cceb9e8282536882370eb4078645713180c6b32b38
gpu5:/tmp/ds4-tp-dspark-smoke2-worker-20260813.log
  sha256=5b0bd4a4a000e58ceec8c1800586752da7c64f7b35354aa5c6b4d409dc920ebe
gpu2:/tmp/ds4-tp-dspark-smoke2-response-20260813.json
  sha256=e10a092b65d3ae28f493f4d409f3cb094fca357982eac59b8a5049fe0545c1c0
gpu2:/tmp/ds4-tp-target-hash-coordinator-20260813.log
  sha256=da984d722b339998c0a55f0120605745b70e62e7f1735f836acc57003764d382
gpu5:/tmp/ds4-tp-target-hash-worker-20260813.log
  sha256=792ccb0bd88301d340cfaf18bc99cace1487f5b9242377b725c9f9cc5bd21486
gpu2:/tmp/ds4-tp-target-hash-response-20260813.json
  sha256=daedf5e6d498d4bc8c3bdea17d4c991c0367de8c2971e9e2ea43e7f2ddafb9e8
```

## 正式单机 DSpark A/B

双机 Pipeline/TP 的高并发实现当前都不能使用正式 DSpark 路径，因此另在 gpu2
使用 Entrpi DS4 v0.5.6.2 做单机 A/B，验证 drafter 对同一模型和 workload 的实际
收益。两组均使用正式 `/usr/local/bin/ds4-server`、`ctx=524288`、同一 base GGUF，
ON 组额外使用：

```text
--no-mtp --dspark /data/models/ds4/DSpark-drafter-Q2K-Q8-0731.gguf
DS4_CONT_DSPARK=1
DS4_DSPARK_MAX_NLIVE=1
```

这是正式启动路径下的 matched-workload A/B，不是严格单变量实验。ON 侧 inspect
还包含 DSpark 专用 profiling/capture 配置，并设置
`DS4_BATCH_FIT_HEADROOM_MB=8192`；因此下述差异只能归于两条正式路径的整体配置，
不能全部归因于单一 DSpark 开关。

每个 case 连续运行 3 次；表中 wall、prefill、decode 均为中位数：

| Case | DSpark OFF wall | DSpark ON wall | Wall 缩短 | OFF / ON decode | OFF / ON prefill |
|---|---:|---:|---:|---:|---:|
| `1 KiB / 128` | 7.276 s | **5.855 s** | 19.5% | 20.9 / **27.4 tok/s** | 270.9 / 269.6 tok/s |
| `8 KiB / 128` | 11.998 s | **10.239 s** | 14.7% | 17.9 / **24.0 tok/s** | 492.0 / 488.4 tok/s |
| `8 KiB / 512` | 33.454 s | **25.583 s** | 23.5% | 17.9 / **24.8 tok/s** | 492.0 / 488.6 tok/s |

9/9 ON 请求均返回目标 token 数；三组输出 SHA-256 分别与 OFF 一致：128-token
cases 为 `fae23d7317d5ac0bd56e16ab4a357862ac4327d64eaed1bd594d1da69439b10d`，
512-token case 为
`d660d192b7e396a825ee6f6cfd6a7b5995ca93fdb2ebd7a2ab4ea135c4adcca0`。
服务端 metrics 记录 `ds4_spec_drafts_total=1835`、`ds4_spec_hits_total=1835`、
`ds4_spec_accept_ratio=1.0000`，日志也持续出现 `CONT_MTP_ACCEPT(DSpark)`，证明
结果不是只加载 drafter 却退化为 target-only。benchmark 窗口内没有外来 GPU
容器运行事件，两个平台 GPU 容器始终保持 paused；DS4 容器 `restart=0`、
`OOMKilled=false`，且无 earlyoom/SIGTERM。

原始结果保存在 gpu2：

```text
/home/nvidia/ds4-benchmarks/20260813-v0562-dspark-ab/off-fresh/
/home/nvidia/ds4-benchmarks/20260813-v0562-dspark-ab/on-fresh/

OFF JSON sha256=ec02de966db5ef905a37ccf4dd768251ccf3f96d1825a646d6d05b36e6209cf6
ON  JSON sha256=644738090cc484cf9bc37d7ac1e83b2f3490e64c668bd6aff761e1e8c1ae4049
ON server.log sha256=fe03a5e7079319f22c14699475c03c63f37b825ba0b8ac648bd2cbe18503b86a
ON SHA256SUMS sha256=fdb16b5c2872fe473d4439b19d2b6ede76930f5cb79a707540f8893337fe7a32
```

该 ON 清单包含 59 个文件，`sha256sum -c` 全部通过。`/stats` 在此版本返回 404，
speculative counters 以 `/metrics` 为准。

这是 deterministic integer-sequence 上的单流 speculative decode 上限，不代表
自然语言 accept rate，也不是双机并发结果。

### 正式单机 DSpark 并发上限

双机正式 DSpark 不受 distributed engine 支持，因此最高并发改在
gpu2 单机上测量。本节仍使用 Entrpi DS4 v0.5.6.2 正式镜像，
但为测量最高性能改用 `ctx=4096`、`DS4_SERVER_COALESCE_MAX_TOKENS=4096`
和 `DS4_CONT_PREFILL_CHUNK=4096`。gpu2 使用 MAXN + `jetson_clocks`，
CPU/GPU/NVD/EMC 分别锁到 `2.601/1.575/1.692/4.266 GHz`。`lzc-earlyoom`
全程保持启用；两个 BGE GPU 容器在计分窗口内精确停止，而不是
pause 后继续占用 UMA。

并发扫描拆成三个独立启动模式：

| 模式 | 关键设置 | 测量含义 |
|---|---|---|
| 生产默认门控 | `DS4_DSPARK_MAX_NLIVE=1` | `banks_live=1` 时使用 DSpark，多 bank 主体自动转 target-only |
| 强制真实并发 DSpark | `DS4_DSPARK_MAX_NLIVE=0` + `DS4_DSPARK_VERIFY_FIT_ROWS=8` | 要求多个 live bank 同时执行 draft/verify |
| DSpark OFF | 不加 `--dspark`，`DS4_CONT_DSPARK=0` | target-only 最高 aggregate 和 matched A/B 基线 |

`1 KiB / 128` 每个有效点连续运行 3 波；所有有效请求均返回
128 token、`finish_reason=length`和同一输出 SHA-256
`fae23d7317d5ac0bd56e16ab4a357862ac4327d64eaed1bd594d1da69439b10d`。

| 模式 | C | 三波 aggregate output | 中位数 | max `banks_live` | Draft/hit 增量 | 结果 |
|---|---:|---|---:|---:|---:|---|
| 默认门控 | 1 | 21.1167, 21.6345, 21.7227 | **21.6345** | 1 | 300/300 | 真实单 bank DSpark |
| 默认门控 | 2 | 18.9968, 18.8931, 19.2157 | **18.9968** | 2 | 0/0 | target-only 主体 |
| 默认门控 | 3 | 22.2637, 22.3785, 22.3913 | **22.3785** | 3 | 0/0 | 9/9 完整，生产默认最高稳定点 |
| 默认门控 | 4 | 23.6466, invalid, invalid | invalid | 4 | 0/0 | 第二波 `cudaMallocAsync` OOM |
| 强制 DSpark | 2 | 20.6547, 20.8648, 20.8773 | **20.8648** | 2 | 564/564 | 6/6 完整，真实两路 speculation |
| 强制 DSpark | 3 | invalid, invalid, invalid | invalid | 3 | - | 首波仅 1/3 完成，`cudaMallocAsync` OOM |
| DSpark OFF | 4 | 23.8626, 23.8866, 23.9150 | **23.8866** | 4 | 0/0 | 12/12 完整 |
| DSpark OFF | 5 | 24.5990, 24.6318, 24.6291 | **24.6291** | 5 | 0/0 | 15/15 完整，本 workload 最高 aggregate |
| DSpark OFF | 6 | invalid, invalid, invalid | invalid | 6 | 0/0 | 首波 `cudaMallocAsync` OOM |

有效最高点的最低 `MemAvailable` 仍为 `20.955-22.330 GiB`，且无
earlyoom/SIGTERM；失败点也未跌破 earlyoom 5% 阈值。这些失败是
CUDA async allocator 容量边界，容器均为 `OOMKilled=false`、退出码 133，
不能用系统尚有可用 UMA 否定。

`8 KiB / 512` 长输出结果如下：

| 模式 | C | 三波 aggregate output | 中位数 | max `banks_live` | Draft/hit 增量 | 结果 |
|---|---:|---|---:|---:|---:|---|
| 默认门控 | 3 | 17.1788, 21.3662, 21.3686 | **21.3662** | 3 | 24/24 | 9/9 完整；只在短暂 `banks_live=1` 窗口投机 |
| 强制 DSpark | 2 | 16.8568, 20.4894, 20.4882 | **20.4882** | 2 | 2294/2294 | 6/6 完整，真实两路 speculation |
| DSpark OFF | 2 | 15.5494, 18.3841, 18.2933 | **18.2933** | 2 | 0/0 | 6/6 完整，matched A/B 基线 |
| DSpark OFF | 4 | 17.9143, 22.6633, 22.7035 | **22.6633** | 4 | 0/0 | 12/12 完整，最高安全 aggregate |
| DSpark OFF | 5 | invalid, invalid, invalid | invalid | 5 | 0/0 | 首波仅 4/5 完成，`cudaMallocAsync` OOM |

强制两路 DSpark 与 fresh OFF 两 bank 基线的长输出中位 aggregate 为
`20.4882` 对 `18.2933 tok/s`，提升约 **12.0%**。默认门控 C=3
虽在首波开始时累计 24 个 draft/hit，但 sampler 只在 `banks_live=1`
时观察到它们增长，第 2/3 波均为 0；因此该结果只称为默认
DSpark 服务 aggregate，不称为三路 simultaneous speculation。

计分前的三组失败尝试不纳入成绩：未显式使用
`/home/nvidia/.local/bin/uv` 的客户端没有发出请求；`ctx=524288`
因可用内存低于 8 GiB admission floor 退化为 serial；BGE 容器仅 pause
时仍占用约 8.6 GiB UMA，导致早期 `ctx=4096` 尝试被 earlyoom 终止。

原始证据保存在 gpu2：

```text
/home/nvidia/ds4-benchmarks/20260813-v0562-dspark-concurrency/
SHA256SUMS: 462 files
SHA256SUMS sha256=e08ff3d3df48f6caeaeadfe7c311e48ad11c6853b3743a701c4a9139ff31e9fe

default C=3 short manifest sha256=2a895753006cf2f65efdedf82ceff7f8d851d62df19ab230c3fe8c2cb4c0f157
default C=3 long  manifest sha256=e02aa628dd8356aa71db35520b611e97a004b67a7669126f5ed3bdfdd4ef006a
forced  C=2 short manifest sha256=36a7e8a269e581e65b39363e5d1a629e8444fa892e7a16ce64e1eeec32619fff
forced  C=2 long  manifest sha256=dd07889a427b1fe2a8bb0347773429d079b4b8b37e35a02837b3d3fe3093e0ac
OFF     C=5 short manifest sha256=041193846f18875a3e0f8e36ee6ac40a39abc4ac3d36c5de41bb00d77c6f889a
OFF     C=4 long  manifest sha256=c90ad1ca9f7fedd5c4ce913098c87f99871ae6246b3950541205f63716f67654
```

## 上游状态与限制

- PR #754 支持 world 2/4 的 CUDA NCCL network TP；每台机器需要完整 GGUF，
  各 rank 应使用相同 commit 和相同模型 bytes。本轮各 rank 使用相同 context；
  协议本身允许不同的非零 context capacity，并按最小值协商。
- `b030961` 的 distributed layer pipeline 只有 `distributed.role == NONE` 才加载
  external support model，distributed speculative 入口也直接退化为单 token，
  因此即使传入 drafter 也会静默运行 target-only。PR #754 的 network TP 则明确
  拒绝 external MTP/DSpark。上面的临时 binary 只用于协议 smoke，不改变正式
  上游能力；高并发结果只能标为 base GGUF target-only，不能称为 DSpark 加速。
- Stock PR #754 的 CUDA leader 在 `count > 1` 时未接通 network batch frame/ACK；
  本报告历史 `ctx=32768` C=4 数据来自旧 ref 临时 patched binary，current-head
  fixed 数据来自另一份 benchmark-only decode-batch glue binary。两者都不
  代表 PR #754 原始 binary 或 stock DS4 的 batching 能力，也不能混成一条曲线。
- 本轮只得到实验性 TP DSpark 非 batched C=1 smoke，没有有效的 DSpark + 双机
  Pipeline 或 batched TP 并发结果；高并发结论只覆盖 base GGUF 和上述明确标记的
  decode-batch patch。
- Stock distributed pipeline 的 `--batched-session > 1` 会让多个 session 重复
  bind 同一 listen port；上游 issue `#784` 仍为 open。本轮 pipeline 并发因此
  使用两个独立完整副本，而不是一个服务内的 resident session batch。
- 单 pipeline endpoint 的 `C=2` 长输出补测（`/tmp/pipeline-single-bench-c2-8k512-20260812.log`，
  SHA-256 `bccb2652af673f9c3f073ace904e1814f633b3ada0b8a47de2465b20f8787b85`）三波
  aggregate 为 `9.1173/9.1500/9.1403 tok/s`，但每波两个请求约 `56/112 s` 交错完成，
  证明是串行 admission/排队，不计入 Pipeline 并行峰值。
- `DS4_CUDA_TP_FAST_ALIGNED_EXPERTS=1` 会改变浮点舍入，本轮未启用。
- 当前传输控制面未认证、未声明 release-stable，只能用于可信直连网络。
- 相关讨论：[`#469`](https://github.com/antirez/ds4/issues/469)、
  [`#651`](https://github.com/antirez/ds4/issues/651)、
  [`#262`](https://github.com/antirez/ds4/issues/262)。

## 清理状态

最终复核时，两机没有运行中的 `ds4-server`、测试容器或 tegrastats；本轮测试端口
和所有本地测试 tunnel 均已关闭，包括 `28562`。收尾时 `docker ps -a | grep ds4`
为空；本轮开始前
曾存在的 stopped 容器 `ds4-tp-coordinator`、`ds4-main-coordinator` 在最终复核时
已经不存在，本轮没有删除它们。模型和已验证镜像保留，未执行 `docker system prune`
或 drop caches。

gpu2 的平台编排在测试期间会重建 BGE 容器。早期双机测试使用
pause guard；单机 DSpark 并发计分时改为精确停止两个 BGE 容器，
并用 stop guard 防止重建后抢占 UMA。收尾已停止 guard，两个原容器
均重新启动并恢复 `healthy`。

两机已使用本轮 MAXN 前保存的 `/tmp/ds4-peak3-gpu{2,5}-restore-20260813.conf`
执行 `jetson_clocks --restore`，并恢复原功耗模式：gpu2 `MAXN, mode 0`，gpu5
`120W, mode 1`。最终只读复核：

| 主机 | CPU policy | GPU min/max | NVD min/max | BWMGR/EMC |
|---|---|---|---|---|
| gpu2 | `972-2601 MHz, schedutil` | `315-1575 MHz` | `315-1692 MHz` | `4266/4266 MHz` |
| gpu5 | `972-2601 MHz, schedutil` | `315-1386 MHz` | `315-1557 MHz` | `4266/4266 MHz` |

CPU、GPU 和 NVD 已恢复动态频率范围；BWMGR/EMC 的实验前快照本身即为
`4266 MHz` min=max。单机 DSpark 并发阶段只重新锁定 gpu2，结束后已用
`/tmp/ds4-dspark-concurrency-clock-snapshot.conf` 再次恢复上表动态范围。
本地早期错误 partial
`/home/xxnuo/DeepSeek-V4-Flash-DSpark-support-0731.gguf`（207,286,272 bytes）
已精确删除；完整校验文件仍保留在下载目录和 gpu2 模型缓存。
