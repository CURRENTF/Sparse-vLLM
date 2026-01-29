import torch


do_sel_layers = [8]


def token_decode_attention_flash_decoding(
    q,
    infer_state,
    q_head_num,
    head_dim,
    cache_k,
    cache_v,
    out=None,
    alloc_tensor_func=torch.empty,
    do_select=False,
    world_size=1,
    tp_rank=0,
):
    from .flash_decoding_stage1 import (
        flash_decode_stage1,
        flash_decode_stage1_with_score,
    )
    from .flash_decoding_stage2 import flash_decode_stage2

    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    calcu_shape1 = (batch_size, q_head_num, head_dim)
    max_len_in_batch = infer_state.max_len_in_batch

    # +++ start hack +++
    assert not infer_state.is_prefill
    if do_select:
        with torch.cuda.nvtx.range("do_select_and_sparse"):
            # assert infer_state.now_layer_idx in do_sel_layers
            req_to_token_indexs = infer_state.req_manager.req_to_token_indexs
            b_seq_len = infer_state.b_seq_len
            b_req_idx = infer_state.b_req_idx
            assert len(b_req_idx.shape) == 1
            infer_state.imp_seq_len = torch.zeros_like(b_seq_len)
            infer_state.max_imp_len_batch = 0

            # +++ triton +++
            o_tensor = (
                alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out
            )
            mid_o = alloc_tensor_func(
                [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, head_dim],
                dtype=torch.float32,
                device="cuda",
            )
            mid_o_logexpsum = alloc_tensor_func(
                [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1],
                dtype=torch.float32,
                device="cuda",
            )
            attn_score = alloc_tensor_func(
                [batch_size, q_head_num, max_len_in_batch],
                dtype=torch.float32,
                device="cuda",
            )

            flash_decode_stage1_with_score(
                q.view(calcu_shape1),
                cache_k,
                cache_v,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_seq_len,
                max_len_in_batch,
                mid_o,
                mid_o_logexpsum,
                attn_score,
                BLOCK_SEQ,
            )
            attn_score = attn_score.max(dim=1).values
            assert batch_size == 1
            for b_cur in range(batch_size):
                real_len = b_seq_len[b_cur]
                num_imp_tokens = min(2048, real_len)
                topk_v, topk_idx = torch.topk(
                    attn_score[b_cur, :real_len], dim=-1, k=num_imp_tokens
                )
                req_id = b_req_idx[b_cur]
                with torch.cuda.nvtx.range("do_4"):
                    torch.index_select(
                        req_to_token_indexs[req_id, :real_len],
                        dim=-1,
                        index=topk_idx,
                        out=infer_state.req_manager.req_to_imp_token_indexs[
                            tp_rank, req_id, :num_imp_tokens
                        ],
                    )
                infer_state.imp_seq_len[b_cur] = num_imp_tokens
                infer_state.max_imp_len_batch = max(
                    infer_state.max_imp_len_batch, num_imp_tokens
                )
            _idx = torch.where(
                infer_state.req_manager.req_to_imp_token_indexs[
                    tp_rank, req_id, :num_imp_tokens
                ]
                >= infer_state.mem_manager.cache_cpu_kv[9].shape[0],
                1,
                0,
            ).cpu()  # 不对，但是只测效率
            for i in range(9, 32):
                assert str(infer_state.mem_manager.cache_cpu_kv[i].device) == "cpu"
                infer_state.imp_cache[i] = torch.index_select(
                    infer_state.mem_manager.cache_cpu_kv[i],
                    dim=0,
                    index=_idx,
                ).to(device="cuda:0", non_blocking=True)

            # +++ torch +++ for debug
            # std_q = q.view(batch_size, q_head_num, head_dim)
            # assert batch_size == 1
            # for b_cur in range(batch_size):
            #     real_len = b_seq_len[b_cur]
            #     # 拿出来算attn map  [len, k_head, dim]
            #     with torch.cuda.nvtx.range("do_1"):
            #         real_cache_k = torch.index_select(
            #             cache_k, dim=0, index=req_to_token_indexs[b_cur, :real_len]
            #         )
            #     # 1, q_heads, 1, 128
            #     real_q = std_q[b_cur].view(1, q_head_num, 1, head_dim)
            #     assert real_q.shape[1] == (32 // world_size), f"{real_q.shape}"
            #     # 1, k_heads, len, 128
            #     real_cache_k = real_cache_k.permute(1, 0, 2)[None, :]
            #     n_rep = real_q.shape[1] // real_cache_k.shape[1]
            #     # print(real_q.shape, real_cache_k.shape)
            #     with torch.cuda.nvtx.range("do_2"):
            #         real_cache_k = (
            #             real_cache_k[:, :, None]
            #             .expand(-1, -1, n_rep, -1, -1)
            #             .reshape(1, q_head_num, -1, head_dim)
            #         )

            #     # print(real_q.shape, real_cache_k.shape)
            #     # 1, 8, 1, len
            #     with torch.cuda.nvtx.range("do_3"):
            #         attn_score_tt = torch.matmul(real_q, real_cache_k.transpose(2, 3))
            #         attn_score_tt = attn_score_tt.max(dim=1).values.squeeze()

            #         print(attn_score)
            #         print(attn_score_tt)
            #         num_imp_tokens = min(2048, attn_score_tt.shape[-1])
            #         topk_v, topk_idx_tt = torch.topk(
            #             attn_score_tt, dim=-1, k=num_imp_tokens
            #         )
            #         req_id = b_req_idx[b_cur]
            #     assert (
            #         torch.sum(torch.eq(topk_idx, topk_idx_tt)) == topk_idx.numel()
            #     ), f"{torch.sum(torch.eq(topk_idx, topk_idx_tt))}\n{topk_idx}\n{topk_idx_tt}"
            #     print(topk_idx_tt[:10])
            #     print(topk_idx[:10])
            #     with torch.cuda.nvtx.range("do_4"):
            #         torch.index_select(
            #             req_to_token_indexs[b_cur, :real_len],
            #             dim=-1,
            #             index=topk_idx,
            #             out=infer_state.req_manager.req_to_imp_token_indexs[
            #                 tp_rank, req_id, :num_imp_tokens
            #             ],
            #         )
            #     infer_state.imp_seq_len[b_cur] = num_imp_tokens
            #     infer_state.max_imp_len_batch = max(
            #         infer_state.max_imp_len_batch, num_imp_tokens
            #     )
            # max_len_in_batch = infer_state.max_imp_len_batch
            # 这里不应该更新 max_len，因为处于do-select，所以应该使用原始内容
    else:
        # assert infer_state.now_layer_idx not in do_sel_layers
        # if infer_state.now_layer_idx < do_sel_layers[0]:
        #     assert infer_state.imp_seq_len is None
        if infer_state.imp_seq_len is not None:
            max_len_in_batch = infer_state.max_imp_len_batch

        o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out
        mid_o = alloc_tensor_func(
            [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, head_dim],
            dtype=torch.float32,
            device="cuda",
        )
        mid_o_logexpsum = alloc_tensor_func(
            [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1],
            dtype=torch.float32,
            device="cuda",
        )

        req_to_imp_token_indexs = infer_state.req_manager.req_to_token_indexs
        if infer_state.imp_seq_len is not None:
            req_to_imp_token_indexs = torch.zeros(
                1, 2048, dtype=torch.int, device=cache_k.device
            )

            cache_k = infer_state.imp_cache[infer_state.now_layer_idx][:, 0:8, :]
            cache_v = infer_state.imp_cache[infer_state.now_layer_idx][:, 8:16, :]

        flash_decode_stage1(
            q.view(calcu_shape1),
            cache_k,
            cache_v,
            req_to_imp_token_indexs,
            infer_state.b_req_idx,
            (
                infer_state.b_seq_len
                if infer_state.imp_seq_len is None
                else infer_state.imp_seq_len
            ),
            max_len_in_batch,
            mid_o,
            mid_o_logexpsum,
            BLOCK_SEQ,
        )

    flash_decode_stage2(
        mid_o,
        mid_o_logexpsum,
        (
            infer_state.b_seq_len
            if do_select or infer_state.imp_seq_len is None
            else infer_state.imp_seq_len
        ),
        o_tensor.view(calcu_shape1),
        BLOCK_SEQ,
    )
    return o_tensor
