# ADR 0004：线性 Chain Prefix Cache

## 状态

已接受。

## 背景

SnapKV、H2O、PyramidKV、R-KV 和 SkipKV 会物理删除 KV position。压缩后，
逻辑 token position 与各 layer 的物理 row length 会发生分离，因此 radix
prefix node 无法描述这些方法的驻留 payload。不过，这些方法仍可受益于常见
的多轮场景：一个会话始终只有一个 continuation writer。

## 决策

`enable_prefix_caching` 仍作为 feature switch；`prefix_cache_mode` 可选
`auto`、`radix` 或 `chain`。Auto 为 vanilla、OmniKV 和 QuEST 解析为
radix，为 SnapKV、H2O、PyramidKV、R-KV 和 SkipKV 解析为 chain。显式的
不兼容组合会在构造 config 时快速失败。

Chain 实现与 `RadixPrefixIndex` 完全独立：

- `ChainCacheIndex` 负责 opaque ID、ACTIVE/IDLE 生命周期、processed-token
  digest、紧凑的 driver-side 逻辑 token 历史、严格的 IDLE-only LRU metadata
  以及有界 tombstone。
- `ChainCacheCoordinator` 只负责逻辑协调。
- Cache manager 负责 KV row、物理 slot、H2O score/cursor state、R-KV
  query、SkipKV sentence state 以及其他所有方法 metadata。
- `RuntimeState` 是 payload 回收入口。

省略、设为 null 或传入空 `chain_id` 时，服务端创建 opaque ID。复用 ID 时，
record 必须为 IDLE，方法/config fingerprint 必须相同，并且输入在持久化的
processed boundary 前必须精确匹配 SHA-256。Rank 0 还会用紧凑 unsigned
32-bit array 保存准确的逻辑 prefix。文本 API 需要原 token identity，因为
BPE prefix 的 decode 后再 encode 通常不会保持 token 稳定；有了保存的 prefix，
服务端只需 tokenize 新增文本。该历史最多为
`max_model_len * max_num_seqs_in_gpu` 个 token，随 chain 回收，并通过
chain-cache token/byte 统计暴露。TP admission validation 和 completion RPC
仍然只传递 token count 与 32-byte digest，因此长 prompt 不会占用固定大小
shared-memory command buffer，worker rank 也不会复制 driver token 历史。

一个 chain 只有一个 ACTIVE writer。正常 EOS 或达到 length 后保留驻留 row，
并转为 IDLE。最后一个 sampled token 尚未执行 forward，因此持久化 boundary
为 `seq.num_tokens - 1`；该 token 属于下一轮 suffix。服务端检测到 text stop
时会使 chain 失效：隐藏 stop text 可能包含已经处理、但不在 client-visible
continuation 中的 token，而通用路径无法回滚已压缩的物理布局。Disconnect、
failure、preemption、cancellation 和 parse failure 同样会使 chain 失效并释放
全部 payload。

LRU 只考虑 IDLE chain，并按 `(last_access, chain_id)` 排序。ACTIVE chain
保持 pinned。Rank 0 通过 TP RPC path 提供准确 victim plan，各 rank 执行并
检查相同生命周期结果。Admission plan 还会在 prefill allocation 前预留逐
layer 物理峰值与 resident row，避免并发排队的 chain 重复承诺同一批 free slot。

## HTTP 与路由契约

Chat Completions、Completions 和 Responses 都接受 `chain_id`。Admission
在构造 streaming response 前完成，因此 chain error 会保留 404/409/410/503
status，而不是在 HTTP 200 后才出现。成功的单 chain 响应通过
`X-SparseVLLM-Chain-ID` 暴露 ID；JSON/SSE object 暴露 `chain_id` 和复用
token usage。未显式传入 `chain_id` 的多 prompt Completions 请求会为每个
prompt 创建独立 chain；各 choice 与 streaming chunk 携带自己的
`chain_id`，响应不会设置只能表达单值的 header 或顶层 ID。因此显式传入
`chain_id` 的 continuation 必须恰好包含一个 prompt。若 batch 中任一
admission 失败，服务端会先取消已完成的部分 admission，再返回错误。

Worker 提供只读 `/v1/chain_cache/routing_match`。对于非空 ID，smart router
并行探测 worker；唯一 IDLE owner 无论负载如何都会胜出。ACTIVE、missing、
tombstoned 和 duplicate ownership 分别映射到 409、404、410 和 500。显式
空 `chain_id` 只会选择支持 chain 的 worker，并在那里创建新 owner。Router
本地不维护 authoritative ownership map。

## 后果

第一版有意不支持 branching、public release endpoint 或 radix compatibility
shim。被遗弃的 chain 由自动 LRU 回收。需要 branching 的应用必须使用支持
radix 的方法，或创建互相独立的 chain 并重新计算各自第一轮。
