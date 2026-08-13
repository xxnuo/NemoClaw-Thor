# DS4 双机推理运行说明

本文记录 Jetson Thor 双机运行 DeepSeek-V4-Flash DS4 的已验证方法：

| 设备 | 地址 | 资源 | 角色 | 层切分 |
|---|---|---:|---|---|
| gpu2 | `192.168.100.10` | 128 GiB | coordinator | `0:28` |
| gpu3 | `192.168.100.20` | 64 GiB | worker | `29:output` |

该切分在两台设备之间通过三条 25G 光口链路工作。API 只在 gpu2 提供，
客户端访问 `gpu2:8050`。

## 已验证版本

已验证成功的是独立 upstream DS4 镜像：

```text
image:  nemoclaw-thor/ds4:main-sm110-dist
commit: b0309611041655f4e45671cfd9c9886aff161406
arch:    sm_110
```

该镜像必须绕过仓库旧版 `/usr/local/bin/ds4-entrypoint`，直接执行
`/usr/local/bin/ds4-server`。旧 entrypoint 会自动加入 upstream main 已经
删除的 `--no-mtp`、`--mem-floor-gb` 和 `--no-update-check` 参数。

仓库的 `v0.5.4-sm110-thor` 双机路径不要使用：实测 prefill 可以完成，但首个
decode token 后 worker 断开，coordinator 报：

```text
distributed coordinator: replaying ... after distributed route failure
distributed coordinator: route incomplete: missing layer 29
```

## 前置条件

两台机器都必须有已验证的镜像。只需在 gpu2 准备一次，然后传到 gpu3：

```bash
# gpu2：确认镜像存在
docker image inspect nemoclaw-thor/ds4:main-sm110-dist >/dev/null

# gpu2：传输镜像到 gpu3
docker save nemoclaw-thor/ds4:main-sm110-dist \
  | ssh gpu3 'docker load'

# 两台机器分别确认镜像 ID
ssh gpu2 'docker image inspect nemoclaw-thor/ds4:main-sm110-dist --format "{{.Id}}"'
ssh gpu3 'docker image inspect nemoclaw-thor/ds4:main-sm110-dist --format "{{.Id}}"'
```

两台机器都执行：

```bash
cd ~/work/NemoClaw-Thor
ip -br addr
test -s "$HOME/thor-hf-cache/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
```

模型 base GGUF 必须在两台机器的相同路径下。DSpark drafter 是仓库 v0.5.5
单机 entrypoint 的配套文件，本 runbook 的 upstream main 直启命令不加载它。
确认链路和端口：

```bash
# gpu2
ping -c 3 192.168.100.20
ss -ltn | grep -E ':8050|:8060' || true

# gpu3
ping -c 3 192.168.100.10
ss -ltn | grep -E ':8051|:8061' || true
```

需要允许：

| 端口 | 监听设备 | 用途 |
|---:|---|---|
| `8060` | gpu2 | coordinator 控制和 worker 注册 |
| `8061` | gpu3 | worker hidden-state 数据连接 |
| `8050` | gpu2 | OpenAI-compatible API |

两台机器使用 `--network=host`，不需要 Docker 端口映射。

## 启动 worker（gpu3）

先启动 worker，再启动 coordinator。以下命令使用已验证的 131072 context；
两台设备共享同一组环境参数，但每台只加载自己的层范围。

```bash
# ssh gpu3
cd ~/work/NemoClaw-Thor

docker rm -f ds4-main-worker 2>/dev/null || true

docker run -d --name ds4-main-worker \
  --runtime=nvidia --network=host --init \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_CTX=131072 \
  -e DS4_HOST=127.0.0.1 \
  -e DS4_PORT=8051 \
  -e DS4_CUDA_VMM_ARENA=0 \
  -e DS4_CUDA_LAYER_GRAPHS=0 \
  -e DS4_CUDA_MOE_GRAPHS=0 \
  -e DS4_CONT_CAPTURE=0 \
  -v "$HOME/thor-hf-cache/ds4:/data/models/ds4" \
  --entrypoint /usr/local/bin/ds4-server \
  nemoclaw-thor/ds4:main-sm110-dist \
  --role worker \
  --cuda \
  -m /data/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  -c 131072 \
  --host 127.0.0.1 --port 8051 \
  --kv-disk-dir /data/models/ds4/kv-cache \
  --kv-disk-space-mb 8192 \
  --layers 29:output \
  --listen 192.168.100.20 8061 \
  --coordinator 192.168.100.10 8060
```

确认 worker 已连接：

```bash
docker logs --tail=80 ds4-main-worker
```

应看到：

```text
distributed worker: connected to coordinator 192.168.100.10:8060
distributed worker: receive prefetch depth 2 enabled
```

worker 初次加载约 26.33 GiB 模型张量，计划总内存约 29.37 GiB。

## 启动 coordinator（gpu2）

```bash
# ssh gpu2
cd ~/work/NemoClaw-Thor

docker rm -f ds4-main-coordinator 2>/dev/null || true

docker run -d --name ds4-main-coordinator \
  --runtime=nvidia --network=host --init \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_CTX=131072 \
  -e DS4_HOST=127.0.0.1 \
  -e DS4_PORT=8050 \
  -e DS4_CUDA_VMM_ARENA=0 \
  -e DS4_CUDA_LAYER_GRAPHS=0 \
  -e DS4_CUDA_MOE_GRAPHS=0 \
  -e DS4_CONT_CAPTURE=0 \
  -v "$HOME/thor-hf-cache/ds4:/data/models/ds4" \
  --entrypoint /usr/local/bin/ds4-server \
  nemoclaw-thor/ds4:main-sm110-dist \
  --role coordinator \
  --cuda \
  -m /data/models/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  -c 131072 \
  --host 127.0.0.1 --port 8050 \
  --kv-disk-dir /data/models/ds4/kv-cache \
  --kv-disk-space-mb 8192 \
  --layers 0:28 \
  --listen 192.168.100.10 8060 \
  --dist-prefill-chunk 4096 \
  --debug
```

确认 coordinator 完成路由：

```bash
docker logs --tail=100 ds4-main-coordinator
```

必须同时看到：

```text
distributed coordinator API: listening on 192.168.100.10:8060
distributed coordinator: complete route ready: local 0:28 -> 192.168.100.20:8061 Q2 29:output
listening on http://127.0.0.1:8050
```

coordinator 初次加载约 54.95 GiB 模型张量，计划总内存约 57.99 GiB。

## API 测试

在 gpu2 执行：

```bash
curl http://127.0.0.1:8050/v1/models

curl -sS -m 120 \
  http://127.0.0.1:8050/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": false
  }'
```

成功响应的判断标准：

- HTTP 状态为 `200`；
- `choices[0].message.content` 有内容；
- coordinator 日志没有 `route failure`、`missing layer 29` 或 worker 被移除；
- 日志出现 `distributed telemetry`，且 `downstream_wait` 通常为 `0.000ms`。

upstream main 不接受 `reasoning_effort: "off"`。使用该字段会返回
`invalid JSON request`；要关闭思考时按当前 binary 的 `--help thinking` 和
API 行为选择支持的值，最稳妥的基础连通性测试是直接省略该字段。

外部客户端使用：

```text
Base URL: http://192.168.100.10:8050/v1
Model:    deepseek-v4-flash
```

当前命令将 API 绑定到 gpu2 的 loopback。需要让其他机器访问时，把
coordinator 的 `--host 127.0.0.1` 改为 `--host 192.168.100.10`；DS4 本身不做
API key 认证，不要绑定到不受信任的网卡。

## 性能和层切分

本机固定 128 token 输出测试结果：

| 切分 | coordinator 规划内存 | worker 规划内存 | decode |
|---|---:|---:|---:|
| `0:28 / 29:output` | 57.99 GiB | 29.37 GiB | 约 10.79 tok/s |
| `0:26 / 27:output` | 54.30 GiB | 33.06 GiB | 约 10.72 tok/s |

因此当前推荐 `0:28 / 29:output`。层数不应按显存容量机械地二等分；gpu2 有
更多 SM，且当前切分已经在两台设备上有足够内存余量。

## 故障排查

### `route incomplete: missing layer 29`

这是 v0.5.4 双机 decode 回归的已复现特征。先确认没有复用旧容器和旧 image：

```bash
ssh gpu2 'docker ps -a --filter name=ds4-main-coordinator'
ssh gpu3 'docker ps -a --filter name=ds4-main-worker'
ssh gpu2 'docker image inspect nemoclaw-thor/ds4:main-sm110-dist --format "{{.Id}}"'
```

不要继续增加 socket timeout、关闭 CUDA graph 或调整显存参数；这些选项在本次
故障中都没有修复首个 decode 断链。

### `unknown option: --no-mtp` / `--mem-floor-gb` / `--no-update-check`

说明仍然走了仓库旧 entrypoint。检查容器是否包含：

```text
--entrypoint /usr/local/bin/ds4-server
```

### `Connection refused`

worker 必须先启动。检查 gpu2 的 8060 和 gpu3 的 8061：

```bash
ssh gpu2 'ss -ltnp | grep 8060'
ssh gpu3 'ss -ltnp | grep 8061'
```

### `Resource temporarily unavailable`

不要只看这一行判断失败。若随后仍有 `complete route ready`，先发送一个最小
请求并观察是否在 decode 阶段出现 `route failure`。旧测试中该错误伴随 worker
连接被反复重建，最终才表现为 `missing layer 29`。

### gpu2 主机内存很高、GPU memory 数值很低

Thor 使用统一内存。模型 mmap 会计入主机 `mem`，只有实际驻留页面才计入 GPU
统计；这是正常现象。以 DS4 日志的 `resident model`、`planned` 和请求是否成功
为准，不要用单独的 `nvidia-smi` 数值判断模型是否只加载了一小部分。

## 停止和清理

停止双机服务：

```bash
ssh gpu2 'docker stop ds4-main-coordinator'
ssh gpu3 'docker stop ds4-main-worker'
```

模型和 KV 缓存不会被删除。停止后若要释放统一内存页，在对应主机执行：

```bash
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
free -g
```

不要为排查 DS4 执行 `docker system prune`，它与本双机服务无关且会删除其他
镜像和缓存。
