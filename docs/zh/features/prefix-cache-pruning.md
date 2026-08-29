# Prefix cache 修剪

Vanilla 和 OmniKV 的 radix prefix cache 支持在不改变逻辑 token 路由和
stable block ID 的前提下，压紧已有空闲树路径的物理 KV payload。QuEST 的
prefix cache 与 offload 继续可用，但 QuEST page 不支持物理修剪。

通过块对齐的半开区间 `[L, R)` 启动维护任务：

```http
POST /v1/prefix_cache/prune
Content-Type: application/json

{
  "token_ids": [1, 2, 3, 4],
  "range_start": 0,
  "range_end": 4,
  "keep_tokens": 2,
  "policy": "snapkv_global"
}
```

接口返回 HTTP 202 和 `prune_id`。随后查询
`GET /v1/prefix_cache/prune/{prune_id}`，直到状态变成 `completed`、
`blocked` 或 `failed`。`snapkv_global` 与 `kvzip_global` 会跨层、跨 head、
跨 TP rank 汇总分数，得到一个确定性的统一 token mask。

只有 `[L, R)` 内所有块均无引用且不存在传输时才会提交修剪。提交后会在
`[L, L + block_size)` 对应的子树根写入可继承的 `quality_degraded` 记录。
设备分配、容量和删除回收均按实际保留 token 槽计量。offload 仍由逻辑块
持有固定 host page，但 D2H/H2D 只传输保留的块内 offset。

请求也可用 `text` 或完整 OpenAI `chat` 请求代替 `token_ids`；`chat` 会复用
服务端的 chat template 和 reasoning 参数渲染逻辑，适合 agent 在每轮结束后精确
定位刚写入的树路径。SnapKV 可设置
`observation_tokens`，KVzip 可设置 `score_chunk_size` 和
`prev_postfix_size`。`allow_recompress` 目前是保留字段并会显式失败，因为
仅依靠已压紧的树无法重新打分已经删除的 KV。
