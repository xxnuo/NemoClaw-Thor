本文记录两台 Jetson AGX Thor T5000 上运行 DeepSeek-V4-Flash-0731 IQ2
混合量化 GGUF 的实际结果。

说明：

- `单流 decode`：服务端生成阶段速度，不含 prefill。
- `单流 E2E`：完成 token 数 ÷ HTTP wall。
- `并发 aggregate`：整波所有完成 token 数 ÷ 整波 wall，包含 prefill、排队和调度。
- `C` 表示并发请求数。

**单流性能**

| 运行条件 | Workload | 服务端 decode | 单流 E2E output | 说明 |
|---|---|---:|---:|---|
| 单机 DS4，DSpark ON | `1 KiB / 128` | **27.4 tok/s** | **约 21.86 tok/s** | 全部配置最高单流 |
| 单机 DS4，DSpark ON | `8 KiB / 512` | **24.8 tok/s** | **约 20.01 tok/s** | 长输出最高单流 |
| 单机 DS4，DSpark OFF | `1 KiB / 128` | 20.9 tok/s | 约 17.59 tok/s | ON 的匹配基线 |
| 单机 DS4，DSpark OFF | `8 KiB / 512` | 17.9 tok/s | 约 15.30 tok/s | ON 提升明显 |
| 双机 Pipeline，MAXN | `1 KiB / 512` | 13.01–13.02 tok/s | **12.0469 tok/s** | 最快双机单流 |
| 双机 Pipeline，常规配置 | `8 KiB / 512` | 12.03–12.05 tok/s | 约 8.83 tok/s | `ctx=131072` |
| 双机 TP，target-only | `8 KiB / 512` | 5.49–5.52 tok/s | 约 4.52 tok/s | 明显慢于 Pipeline |
| 实验性双机 TP DSpark | `29 token / 64` | 1.83 tok/s | 1.7611 tok/s | 反而慢于 target-only 的 4.4988 |

**并发总吞吐**

| 方式 | Workload | 峰值 aggregate | 最高安全并发 | 更高边界或说明 |
|---|---|---:|---:|---|
| 单机 DSpark OFF | `1 KiB / 128` | **24.6291 tok/s，C=5** | C=5 | C=6 CUDA OOM；全场最高总吞吐 |
| 单机 DSpark OFF | `8 KiB / 512` | **22.6633 tok/s，C=4** | C=4 | C=5 CUDA OOM |
| 默认 DSpark 门控 | `1 KiB / 128` | 22.3785 tok/s，C=3 | C=3 | 多 bank 时主体为 target-only；C=4 OOM |
| 默认 DSpark 门控 | `8 KiB / 512` | 21.3662 tok/s，C=3 | C=3 | 只在短暂单 bank 阶段投机 |
| 强制真实并发 DSpark | `1 KiB / 128` | **20.8648 tok/s，C=2** | C=2 | C=3 CUDA OOM |
| 强制真实并发 DSpark | `8 KiB / 512` | **20.4882 tok/s，C=2** | C=2 | 相对匹配 OFF C=2 的 18.2933 提升 12.0% |
| 双机 Pipeline，两副本 | `1 KiB / 512` | **19.6329 tok/s，C=2** | C=2 | 双机最高；第三副本内存预算不足，未启动 |
| 双机 Pipeline，两副本 | `8 KiB / 512` | 8.6857 tok/s，C=2 | C=2 | 更长 prompt 的端到端结果 |
| 双机 TP fixed batch | `1 KiB / 128` | **4.8279 tok/s，C=2** | C=14：3.3079 tok/s | C=15 为 2.4236，但触发 gpu5 earlyoom |
| 双机 TP patched 长输出 | `8 KiB / 512` | 3.9680 tok/s，C=1 | C=4：3.9592 tok/s | 历史 benchmark-only patch |

最终排名：

| 指标 | 最佳结果 |
|---|---|
| 最高单流服务端 decode | 单机 DSpark ON：**27.4 tok/s** |
| 最高单流端到端输出 | 单机 DSpark ON：**约 21.86 tok/s** |
| 最高并发总吞吐 | 单机 DSpark OFF，C=5：**24.6291 tok/s** |
| 最高真实并发 DSpark | C=2：短输出 **20.8648**，长输出 **20.4882 tok/s** |
| 最高双机总吞吐 | Pipeline 两副本 C=2：**19.6329 tok/s** |
| TP 最高安全并发 | C=14：**3.3079 tok/s** |

DSpark 提升单流和相同 C=2 下的吞吐，但会占用额外显存，所以追求绝对总吞吐时，关闭 DSpark 并提高到 C=4/5 反而更快。完整数据见[性能报告](/home/xxnuo/projects/work/NemoClaw-Thor/serving/docs/DS4-GPU2-GPU5-PARALLEL-BENCHMARK-20260812.md:13)。