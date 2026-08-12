# DS4 gpu2/gpu5 并行实测

验证日期：2026-08-11 至 2026-08-12。

本文记录两台 Jetson AGX Thor T5000 上运行 DeepSeek-V4-Flash-0731 IQ2
混合量化 GGUF 的实际结果。测试覆盖单机冷启动、双机流水线并行、双机 NCCL
张量并行和双请求 admission。prefill/decode 吞吐来自本次容器日志，HTTP wall
time 和并发 aggregate throughput 来自仓库现有 benchmark 脚本；没有把历史
gpu2/gpu3 的 128+64 GiB 数据混入本次 gpu2/gpu5 结论。

## 结论

- 在本轮已完成的 8 KiB 短请求矩阵中，`gpu2 0:20 / gpu5 21:output`
  流水线的三次 wall time 均值最低，且两组各 3 次的波动最小：`8 KiB / 128`
  约 26.1 秒，`8 KiB / 512` 约 58.0 秒。
- NCCL TP 在两台对称 T5000 上达到 ready 并完成 8 KiB 短请求矩阵；decode 约
  `5.4-5.5 tok/s`，balanced pipeline 约 `12.03-12.08 tok/s`。两条路径使用
  不同 DS4 commit，不是只改变并行方式的单变量对照。
- balanced pipeline 的单个 21K-token 请求可完成，但紧接的第二个长请求未进入
  `prompt start`；本轮不能确认随后出现的 data connection error 与卡住存在因果
  关系，也不能称为长请求连续稳定。
- 单机 upstream-main、131072 context 冷启动在约 3 分钟观察窗口内未监听 API，
  进程无 OOM/Xid，但占用约 107 GiB；本轮没有得到可比较的单机 HTTP 数据。

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

本轮双机路径只使用 base GGUF，不加载 DSpark drafter。

| 路径 | 镜像 | DS4 ref | image ID |
|---|---|---|---|
| layer pipeline | `nemoclaw-thor/ds4:main-sm110-dist` | `b0309611041655f4e45671cfd9c9886aff161406` | `sha256:b72249448e072f3f0dff07e53a49e6cb257e1b24126b8c05155abee95fc63fe5` |
| network TP | `nemoclaw-thor/ds4:pr754-sm110-tp` | `d6e64adaa7cd3e16001bc2090e27b76c618a440a` | `sha256:f25d9eae9d93f82218038745ec04f6deb9c1c807e4860a899f1cdf0115a046e6` |

两个 image ID 在 gpu2/gpu5 一致。两个镜像都必须绕过旧 entrypoint，显式执行
`/usr/local/bin/ds4-server`。

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

并发探针使用：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:18350/v1 \
  --label gpu2-gpu5-pipeline-concurrency-20260811 \
  --concurrency 2 --waves 1 --prompt-kib 8 --output 32 --timeout 1800
```

## 流水线并行

### 对称切分：`0:20 / 21:output`

这是两台同型 T5000 的主要结果。coordinator 计划内存 43.24 GiB，worker
计划内存 44.12 GiB；route 为：

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

TP 使用上游开放 PR
[`antirez/ds4#754`](https://github.com/antirez/ds4/pull/754) 的 head
`d6e64adaa7cd3e16001bc2090e27b76c618a440a`。截至 2026-08-12，该 PR
仍未合入 `main`，GitHub 报告 `mergeable_state=dirty`。上游 `main` 已有两机
Metal TP 和单机多 GPU CUDA TP，但没有该 PR 提供的跨节点 CUDA NCCL、world 2/4
network TP 路径。

本轮两个 rank 均使用：

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

## 上游状态与限制

- PR #754 支持 world 2/4 的 CUDA NCCL network TP；每台机器需要完整 GGUF，
  各 rank 应使用相同 commit 和相同模型 bytes。本轮各 rank 使用相同 context；
  协议本身允许不同的非零 context capacity，并按最小值协商。
- TP 不支持 `--layers`、DSpark/MTP 或 SSD streaming。
- `DS4_CUDA_TP_FAST_ALIGNED_EXPERTS=1` 会改变浮点舍入，本轮未启用。
- 当前传输控制面未认证、未声明 release-stable，只能用于可信直连网络。
- 相关讨论：[`#469`](https://github.com/antirez/ds4/issues/469)、
  [`#651`](https://github.com/antirez/ds4/issues/651)、
  [`#262`](https://github.com/antirez/ds4/issues/262)。

## 清理状态

所有本轮命名为 `ds4-gpu*-202608*` 的测试容器均已精确删除；收尾复核时，
gpu2/gpu5 的测试端口 `8150/8160`、`8250/8260`、`8350/8360`、`8450` 均无
监听，也没有运行中的 `ds4-server`。gpu2 仍有两个本轮开始前已有的 stopped
容器 `ds4-tp-coordinator` 和 `ds4-main-coordinator`，未擅自删除。模型和已验证
镜像保留在 gpu5，便于后续复测；未执行 `docker system prune` 或 drop caches。
