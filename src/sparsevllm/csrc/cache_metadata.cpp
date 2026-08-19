#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <stdexcept>

// Fast decode metadata preparation for Standard / Dense CacheManager
void prepare_decode_metadata_standard_cpu(
    const std::vector<int64_t>& input_tokens,
    const std::vector<int64_t>& positions,
    const std::vector<int32_t>& row_indices,
    at::Tensor& row_seq_lens,               // 1D int32 [MAX_ROWS] CPU/pinned
    at::Tensor& free_slots_stack,           // 1D int32 [TOTAL_SLOTS] CPU/pinned
    int64_t num_free_slots,
    at::Tensor& buffer_req_to_token_slots,  // 2D int32 [MAX_ROWS, MAX_LEN] CPU/pinned
    at::Tensor& out_input_ids,              // 1D int64 [batch_size] CPU/pinned
    at::Tensor& out_positions,              // 1D int64 [batch_size] CPU/pinned
    at::Tensor& out_req_indices,            // 1D int32 [batch_size] CPU/pinned
    at::Tensor& out_context_lens,           // 1D int32 [batch_size] CPU/pinned
    at::Tensor& out_slot_mapping            // 1D int32 [batch_size] CPU/pinned
) {
    const int64_t batch_size = input_tokens.size();
    if (batch_size == 0) return;

    if (num_free_slots < batch_size) {
        throw std::runtime_error("Out of KV cache slots in C++ prepare_decode");
    }

    int64_t* out_in_ptr = out_input_ids.data_ptr<int64_t>();
    int64_t* out_pos_ptr = out_positions.data_ptr<int64_t>();
    int32_t* out_req_ptr = out_req_indices.data_ptr<int32_t>();
    int32_t* out_ctx_ptr = out_context_lens.data_ptr<int32_t>();
    int32_t* out_slot_ptr = out_slot_mapping.data_ptr<int32_t>();

    int32_t* row_lens_ptr = row_seq_lens.data_ptr<int32_t>();
    int32_t* free_stack_ptr = free_slots_stack.data_ptr<int32_t>();
    int32_t* req_slots_ptr = buffer_req_to_token_slots.data_ptr<int32_t>();
    const int64_t max_len = buffer_req_to_token_slots.size(1);

    const int64_t stack_ptr = num_free_slots - batch_size;

    for (int64_t b = 0; b < batch_size; ++b) {
        out_in_ptr[b] = input_tokens[b];
        out_pos_ptr[b] = positions[b];

        const int32_t row = row_indices[b];
        out_req_ptr[b] = row;

        const int32_t cur_len = row_lens_ptr[row];
        out_ctx_ptr[b] = cur_len;

        const int32_t slot = free_stack_ptr[stack_ptr + b];
        out_slot_ptr[b] = slot;

        // Record slot into buffer table & increment sequence length
        if (cur_len < max_len) {
            req_slots_ptr[row * max_len + cur_len] = slot;
        }
        row_lens_ptr[row] = cur_len + 1;
    }
}

// Fast multi-layer decode metadata preparation for Layer-wise / SnapKV CacheManager
void prepare_decode_metadata_layered_cpu(
    const std::vector<int64_t>& input_tokens,
    const std::vector<int64_t>& positions,
    const std::vector<std::vector<int32_t>>& layer_row_indices, // [num_layers, batch_size]
    at::Tensor& layers_row_seq_lens,                            // 2D int32 [num_layers, MAX_ROWS]
    at::Tensor& layers_free_slots_stack,                        // 2D int32 [num_layers, TOTAL_SLOTS]
    const std::vector<int64_t>& layers_num_free_slots,          // [num_layers]
    at::Tensor& layers_buffer_req_to_token_slots,               // 3D int32 [num_layers, MAX_ROWS, MAX_LEN]
    at::Tensor& out_input_ids,                                  // 1D int64 [batch_size]
    at::Tensor& out_positions,                                  // 1D int64 [batch_size]
    at::Tensor& out_layers_req_indices,                         // 2D int32 [num_layers, batch_size]
    at::Tensor& out_layers_context_lens,                        // 2D int32 [num_layers, batch_size]
    at::Tensor& out_layers_slot_mapping                         // 2D int32 [num_layers, batch_size]
) {
    const int64_t batch_size = input_tokens.size();
    if (batch_size == 0) return;

    const int64_t num_layers = layer_row_indices.size();

    // 1. Copy input tokens and positions
    int64_t* out_in_ptr = out_input_ids.data_ptr<int64_t>();
    int64_t* out_pos_ptr = out_positions.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        out_in_ptr[b] = input_tokens[b];
        out_pos_ptr[b] = positions[b];
    }

    const int64_t max_rows = layers_row_seq_lens.size(1);
    const int64_t total_slots = layers_free_slots_stack.size(1);
    const int64_t max_len = layers_buffer_req_to_token_slots.size(2);

    int32_t* row_lens_base = layers_row_seq_lens.data_ptr<int32_t>();
    int32_t* free_stack_base = layers_free_slots_stack.data_ptr<int32_t>();
    int32_t* req_slots_base = layers_buffer_req_to_token_slots.data_ptr<int32_t>();

    int32_t* out_req_base = out_layers_req_indices.data_ptr<int32_t>();
    int32_t* out_ctx_base = out_layers_context_lens.data_ptr<int32_t>();
    int32_t* out_slot_base = out_layers_slot_mapping.data_ptr<int32_t>();

    // 2. Process all layers in continuous memory
    for (int64_t l = 0; l < num_layers; ++l) {
        const int64_t num_free = layers_num_free_slots[l];
        if (num_free < batch_size) {
            throw std::runtime_error("Out of KV cache slots in layered C++ prepare_decode");
        }
        const int64_t stack_ptr = num_free - batch_size;

        int32_t* row_lens_l = row_lens_base + l * max_rows;
        int32_t* free_stack_l = free_stack_base + l * total_slots;
        int32_t* req_slots_l = req_slots_base + l * max_rows * max_len;

        int32_t* out_req_l = out_req_base + l * batch_size;
        int32_t* out_ctx_l = out_ctx_base + l * batch_size;
        int32_t* out_slot_l = out_slot_base + l * batch_size;

        const auto& row_indices_l = layer_row_indices[l];

        for (int64_t b = 0; b < batch_size; ++b) {
            const int32_t row = row_indices_l[b];
            out_req_l[b] = row;

            const int32_t cur_len = row_lens_l[row];
            out_ctx_l[b] = cur_len;

            const int32_t slot = free_stack_l[stack_ptr + b];
            out_slot_l[b] = slot;

            if (cur_len < max_len) {
                req_slots_l[row * max_len + cur_len] = slot;
            }
            row_lens_l[row] = cur_len + 1;
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "prepare_decode_metadata_standard_cpu",
        &prepare_decode_metadata_standard_cpu,
        "Fast Standard decode metadata preparation"
    );
    m.def(
        "prepare_decode_metadata_layered_cpu",
        &prepare_decode_metadata_layered_cpu,
        "Fast Multi-Layer / SnapKV decode metadata preparation"
    );
}
