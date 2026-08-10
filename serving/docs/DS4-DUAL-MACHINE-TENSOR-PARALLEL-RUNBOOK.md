# DS4 双机张量并行运行说明

本文记录在两台 Jetson AGX Thor 上运行 DeepSeek-V4-Flash DS4 网络张量并行
的实测方法。它使用上游 PR
[`antirez/ds4#754`](https://github.com/antirez/ds4/pull/754)，不是
[`DS4-DUAL-MACHINE-RUNBOOK.md`](DS4-DUAL-MACHINE-RUNBOOK.md) 中按层切分的
流水线并行。

验证日期：2026-08-10。

## 结论

| 项目 | 本文配置 |
|---|---|
| 并行方式 | 两 rank NCCL tensor parallelism |
| rank 0 | gpu2，128 GiB，`192.168.100.10` |
| rank 1 | gpu3，64 GiB，`192.168.100.20` |
| context | 32768 tokens |
| API | gpu2 `http://127.0.0.1:8050/v1` |
| 模型 | DeepSeek-V4-Flash 0731 Q2 tensor mix |

两个 rank 都执行全部层、保存完整 KV cache，并在同一个 token 上同步工作。
不要传 `--layers`。PR 会把 256 个 routed experts 平分为两组，并切分支持的
attention、dense FFN、shared-expert 和 output-row decode 工作：

```text
rank 0: experts 0..127
rank 1: experts 128..255
```

实测每个 rank 映射 `44.48 GiB` resident model，并额外建立 `6.15 GiB`
aligned dense artifacts。gpu3 的 64 GiB 容量决定了本配置必须使用
`DS4_CUDA_DIRECT_MODEL=1`，并将 context 限制为 32768。PR 原本验证的是
两台或四台 128 GiB DGX Spark，并以保留 32 GiB headroom 为设计目标；
128+64 GiB 是本文额外验证的紧内存配置。

## 已验证版本

```text
PR:       antirez/ds4#754
PR state: open
head:     d6e64adaa7cd3e16001bc2090e27b76c618a440a
fork:     shankinson/ds4
branch:   agent/cuda-network-ep-tp
CUDA:     sm_110
NCCL:     2.28.3
image:    nemoclaw-thor/ds4:pr754-sm110-tp
image ID: sha256:f25d9eae9d93f82218038745ec04f6deb9c1c807e4860a899f1cdf0115a046e6
```

镜像内 `/etc/ds4-build.txt` 应为：

```text
ds4_ref=d6e64adaa7cd3e16001bc2090e27b76c618a440a
cuda_arch=sm_110
profile=pr754-network-tp-nccl
```

两台机器必须使用相同 image ID 和相同 GGUF。网络协议未认证、未加密且尚未
声明 release-stable，只能用于可信主机和可信链路。

## 构建镜像

以下步骤只需在 gpu2 执行一次。运行镜像已有 NCCL 2.28.3 runtime，但没有
开发头文件；编译时使用与 runtime 完全匹配的 NVIDIA NCCL header。

```bash
ssh gpu2
cd ~/work/NemoClaw-Thor

build_dir="$(mktemp -d /tmp/ds4-pr754-build.XXXXXX)"
git clone https://github.com/shankinson/ds4.git "$build_dir/src"
git -C "$build_dir/src" checkout d6e64adaa7cd3e16001bc2090e27b76c618a440a

mkdir -p "$build_dir/nccl"
curl -L \
  https://raw.githubusercontent.com/NVIDIA/nccl/v2.28.3-1/src/nccl.h.in \
  -o "$build_dir/nccl/nccl.h.in"
sed \
  -e 's/${nccl:Major}/2/g' \
  -e 's/${nccl:Minor}/28/g' \
  -e 's/${nccl:Patch}/3/g' \
  -e 's/${nccl:Suffix}//g' \
  -e 's/${nccl:Version}/22803/g' \
  "$build_dir/nccl/nccl.h.in" > "$build_dir/nccl/nccl.h"

make -C "$build_dir/src" -B \
  ds4 ds4-server ds4-bench ds4-eval ds4-agent \
  CUDA_ARCH=sm_110 \
  NCCL_INCLUDE_DIR="$build_dir/nccl"
```

编译输出必须包含 `DS4_CUDA_HAVE_NCCL=1`。以下步骤基于已验证的
`main-sm110-dist` 运行镜像替换五个 frontend binary：

```bash
stage=ds4-pr754-stage
docker rm -f "$stage" 2>/dev/null || true
docker create --name "$stage" nemoclaw-thor/ds4:main-sm110-dist

for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
  docker cp "$build_dir/src/$bin" "$stage:/usr/local/bin/$bin"
done

printf '%s\n' \
  'ds4_ref=d6e64adaa7cd3e16001bc2090e27b76c618a440a' \
  'cuda_arch=sm_110' \
  'profile=pr754-network-tp-nccl' \
  > "$build_dir/ds4-build.txt"
docker cp "$build_dir/ds4-build.txt" "$stage:/etc/ds4-build.txt"

docker commit "$stage" nemoclaw-thor/ds4:pr754-sm110-tp
docker rm "$stage"
```

验证并同步到 gpu3：

```bash
docker run --rm --entrypoint /bin/sh \
  nemoclaw-thor/ds4:pr754-sm110-tp \
  -c 'cat /etc/ds4-build.txt'

docker save nemoclaw-thor/ds4:pr754-sm110-tp \
  | ssh gpu3 'docker load'

ssh gpu2 'docker image inspect nemoclaw-thor/ds4:pr754-sm110-tp --format "{{.Id}}"'
ssh gpu3 'docker image inspect nemoclaw-thor/ds4:pr754-sm110-tp --format "{{.Id}}"'
```

## 网络检查

已验证的三条 25G 链路如下，六个接口的 MTU 都是 8966：

| link | gpu2 | gpu3 |
|---|---|---|
| `mgbe0_0` | `192.168.100.10/24` | `192.168.100.20/24` |
| `mgbe1_0` | `192.168.101.10/24` | `192.168.101.20/24` |
| `mgbe2_0` | `192.168.102.10/24` | `192.168.102.20/24` |

两台机器分别检查：

```bash
ip -br addr show mgbe0_0
ip -br addr show mgbe1_0
ip -br addr show mgbe2_0
ip -d link show mgbe0_0 | grep mtu
ip -d link show mgbe1_0 | grep mtu
ip -d link show mgbe2_0 | grep mtu
```

NCCL Socket 可以发现三张网卡，但本次两 rank DeepSeek TP 的小粒度 collective
实际只在 `mgbe0_0` 建立数据连接。强制 6 channels 仍只使用
`NET/Socket/0`，且短请求 decode 由 `1.95 tok/s` 降至 `1.19 tok/s`，因此
正式配置保留 NCCL 默认 channel 数。三链路聚合吞吐不能直接等同于本工作负载
的有效带宽。

## 启动 rank 1（gpu3）

先启动 worker。旧镜像 entrypoint 会注入 PR 已删除的 `--no-mtp`，所以必须
显式使用 `/usr/local/bin/ds4-server`。

```bash
ssh gpu3
cd ~/work/NemoClaw-Thor

docker rm -f ds4-tp-worker 2>/dev/null || true

docker run -d --name ds4-tp-worker \
  --runtime=nvidia --network=host --init \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_CTX=32768 \
  -e DS4_CUDA_DIRECT_MODEL=1 \
  -e DS4_CUDA_MOE_GRAPHS=0 \
  -e DS4_CUDA_VMM_ARENA=0 \
  -e DS4_CUDA_LAYER_GRAPHS=0 \
  -e DS4_CONT_CAPTURE=0 \
  -e DS4_TP_TIMEOUT_SEC=300 \
  -e DS4_NCCL_LIBRARY=/lib/aarch64-linux-gnu/libnccl.so.2 \
  -e DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096 \
  -e NCCL_SOCKET_IFNAME=mgbe \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -v "$HOME/thor-hf-cache/ds4:/data/models/ds4" \
  --entrypoint /usr/local/bin/ds4-server \
  nemoclaw-thor/ds4:pr754-sm110-tp \
  --role worker \
  --cuda \
  -m /data/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  -c 32768 \
  --host 127.0.0.1 --port 8051 \
  --tensor-parallel \
  --tensor-parallel-world 2 \
  --tensor-parallel-rank 1 \
  --coordinator 192.168.100.10 8060 \
  --transport nccl
```

## 启动 rank 0（gpu2）

```bash
ssh gpu2
cd ~/work/NemoClaw-Thor

docker rm -f ds4-tp-coordinator 2>/dev/null || true

docker run -d --name ds4-tp-coordinator \
  --runtime=nvidia --network=host --init \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_CTX=32768 \
  -e DS4_CUDA_DIRECT_MODEL=1 \
  -e DS4_CUDA_MOE_GRAPHS=0 \
  -e DS4_CUDA_VMM_ARENA=0 \
  -e DS4_CUDA_LAYER_GRAPHS=0 \
  -e DS4_CONT_CAPTURE=0 \
  -e DS4_TP_TIMEOUT_SEC=300 \
  -e DS4_NCCL_LIBRARY=/lib/aarch64-linux-gnu/libnccl.so.2 \
  -e DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096 \
  -e NCCL_SOCKET_IFNAME=mgbe \
  -e NCCL_SOCKET_FAMILY=AF_INET \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=WARN \
  -v "$HOME/thor-hf-cache/ds4:/data/models/ds4" \
  --entrypoint /usr/local/bin/ds4-server \
  nemoclaw-thor/ds4:pr754-sm110-tp \
  --role coordinator \
  --cuda \
  -m /data/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  -c 32768 \
  --host 127.0.0.1 --port 8050 \
  --tensor-parallel \
  --tensor-parallel-world 2 \
  --listen 192.168.100.10 8060 \
  --transport nccl
```

两台机器首次冷启动都要构建 `42.43 GiB` shard-local aligned artifacts。
本次 gpu2 warm 约 39 秒，gpu3 约 176 秒。

## 成功标记

gpu2：

```text
expert shard (rank 0/2): mapping 219 spans, 44.48 GiB of 80.76 GiB
ds4-tp: worker rank 1 connected (1/1)
ds4-tp: rank 0/2 ready, transport=nccl
ds4: NCCL collective ready: rank 0/2 device=0 version=2.28.3
tensor parallelism bound: rank 0/2, NCCL transport
listening on http://127.0.0.1:8050
```

gpu3：

```text
expert shard (rank 1/2): mapping 219 spans, 44.48 GiB of 80.76 GiB
ds4-tp: rank 1/2 ready, transport=nccl
ds4: NCCL collective ready: rank 1/2 device=0 version=2.28.3
tensor parallelism bound: rank 1/2, NCCL transport
tp worker ready for mirrored sessions
```

32768 context 的计划内存应为：

```text
KV 0.78 GiB + buffers 0.25 GiB + resident model 44.48 GiB
+ aligned dense artifacts 6.15 GiB = 51.65 GiB planned
```

## API smoke

在 gpu2 执行：

```bash
curl -sS -m 300 \
  http://127.0.0.1:8050/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 32,
    "temperature": 0,
    "reasoning_effort": "none",
    "stream": false
  }'
```

成功标准：HTTP 200、有非空 `choices[0].message.content`，两容器继续运行，
日志没有 rank disconnect、collective failure 或 session failure。

## Benchmark 方法

benchmark 复用仓库现有脚本，不在运行中的模型旁再加载一份 engine-side
`ds4-bench`。在本地工作区建立到 gpu2 loopback API 的 SSH tunnel：

```bash
ssh -N -L 18050:127.0.0.1:8050 gpu2
```

HTTP 矩阵：

```bash
uv run python serving/benchmarks/bench-ds4-http.py \
  --base-url http://127.0.0.1:18050/v1 \
  --label dual-tp-pr754-exact-20260810 \
  --case 8:128 \
  --case 8:512 \
  --case 72:128 \
  --repeats 3 \
  --timeout 1800 \
  --output-json /tmp/ds4-dual-tp-exact-20260810.json
```

PR binary 的 HTTP 响应没有 `timings` 字段，因此脚本 JSON 中的
`prefill_tok_s`、`decode_tok_s` 和 `ttft_ms` 为 0。实际 prefill/decode
从 coordinator 日志中的 `prompt done`、`decoding ... avg=` 和
`finish=length` 提取；HTTP JSON 的 wall time 仍然有效。

并发 probe：

```bash
uv run python serving/benchmarks/bench-ds4-concurrency.py \
  --base-url http://127.0.0.1:18050/v1 \
  --label dual-tp-pr754-exact-concurrency-20260810 \
  --concurrency 2 \
  --waves 1 \
  --prompt-kib 8 \
  --output 32 \
  --timeout 1800
```

它测量两个客户端同时到达时的完成时间和 aggregate output throughput，
不代表单请求 decode 速度，也不证明服务器存在两个独立 resident session。
本次为小并发安全样本（`8 KiB / 32`），避免在 64 GiB rank 上同时申请两个长
context session。

## 实测结果

### 张量并行

| 模式 | Prompt | Output | Repeats | Median prefill | Median decode | Median wall |
|---|---:|---:|---:|---:|---:|---:|
| TP exact | 8 KiB | 128 | 3 | 86.6 tok/s (27.90s) | 4.14 tok/s | 59.13s |
| TP exact | 8 KiB | 512 | 3 | 96.2 tok/s (25.07s) | 3.68 tok/s | 163.78s |
| TP exact | 72 KiB | 128 | 3 | 97.5 tok/s (215.99s) | 3.47 tok/s | 252.91s |

PR 的可选 `DS4_CUDA_TP_FAST_ALIGNED_EXPERTS=1` 会使用 aligned MMQ
representation，改变浮点舍入，不是默认 byte-identical parity 路径。本机
单次 `8 KiB / 128` 对照如下。该样本使用 131072 context，主要用于判断开关
方向，不是与最终 32768 exact 配置的严格单变量对照：

| 模式 | Prompt tokens | Output | Prefill | Decode | Wall |
|---|---:|---:|---:|---:|---:|
| TP fast aligned | 2415 | 128 | 132.6 tok/s | 3.02 tok/s | 60.55 s |

### 并发样本

| Concurrency | Prompt | Output | Waves | Wall | Aggregate output |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 KiB | 32 | 1 | 73.48s | 0.87 tok/s |

coordinator 日志显示两个请求分别在 `39.79s` 和 `33.65s` 完成，第二个请求
在第一个完成后才开始；因此该 binary 的默认 HTTP 路径在本配置下是串行
admission，不是两个 session 的并行吞吐。

### 与流水线并行比较

同一模型、同一 benchmark corpus 的 upstream-main 流水线基线为
`gpu2 0:28 / gpu3 29:output`：

| 模式 | Prompt | Output | Median prompt tokens | Median prefill | Median decode | Median wall |
|---|---:|---:|---:|---:|---:|---:|
| Pipeline | 8 KiB | 128 | 2414 | 140.1 tok/s | 10.65 tok/s | 29.56 s |
| Pipeline | 8 KiB | 512 | 2413 | 140.1 tok/s | 10.61 tok/s | 65.74 s |
| Pipeline | 72 KiB | 128 | 21052 | 258.2 tok/s | 9.25 tok/s | 96.45 s |

流水线数据来自 `/tmp/ds4-dual-main-20260810.json` 和
`/tmp/ds4-dual-main-coordinator.log`。两种并行方式使用不同 DS4 commit；
表格用于当前硬件的运行决策，不是严格的单变量算法比较。

当前 128+64 GiB、25G Socket 条件下，pipeline 的 wall time 比 TP 快约
`2.0x` 到 `2.6x`，decode 快约 `2.6x` 到 `2.9x`。TP 的价值是验证上游
network tensor-parallel 机制和平均分片所有权，不是替代这套异构设备上的
pipeline 生产配置。

## 已排除的配置

### 不要使用 `--layers`

`--tensor-parallel` 和 pipeline layer roles 互斥。TP 中每个 rank 都执行完整
层图并保存完整 KV cache，权重所有权由 PR 内部决定。

### 不要使用旧 entrypoint

症状：

```text
unknown option: --no-mtp
```

修复：显式传入：

```text
--entrypoint /usr/local/bin/ds4-server
```

### 64 GiB rank 必须使用 direct model

不设置 `DS4_CUDA_DIRECT_MODEL=1` 时，启动会额外准备约 8.20 GiB tensor-span
cache。gpu3 会在 session create 前后出现 CUDA `NV_ERR_NO_MEMORY`，进程可能
以 137 退出。单独设置 `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` 不可用：未覆盖完整
optional cache 时 PR 会将启动视为失败。

### 131072 context 不是推荐值

131072 context 曾成功完成 HTTP 请求，但每 rank 计划内存为 `53.66 GiB`，
gpu3 在 session allocation 和首请求期间出现多次 `NV_ERR_NO_MEMORY`。32768
context 将计划内存降至 `51.65 GiB`，且覆盖本文最大 21054-token prompt。
它不能完全消除 gpu3 驱动在紧内存 session 分配时的短暂
`NV_ERR_NO_MEMORY` 重试；本次 benchmark 中 rank 未退出、请求均成功。

### 不要强制 6 NCCL channels

设置 `NCCL_MIN_NCHANNELS=6`、`NCCL_MAX_NCHANNELS=6` 后日志确实显示
`6 coll channels`，但实际连接仍为 `NET/Socket/0`，另外两条网卡计数不变，
短 decode 从约 `1.95 tok/s` 降至 `1.19 tok/s`。

## 停止和恢复

```bash
ssh gpu2 'docker rm -f ds4-tp-coordinator'
ssh gpu3 'docker rm -f ds4-tp-worker'
```

如需释放 UMA page cache，在两个容器停止后分别执行：

```bash
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
free -h
```

不要在容器运行时 drop caches，也不要用 `docker system prune` 代替精确停止。
