import torch
from deltakv.modeling import LlamaSnapKVForCausalLM, KVLlamaConfig

def test_init():
    config = KVLlamaConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        snapkv_num_full_layers=1,
        snapkv_window_size=4,
        tail_token_size=16,
        num_sink_tokens=8,
        num_top_tokens=32
    )
    model = LlamaSnapKVForCausalLM(config)
    print("Model initialized successfully!")
    
    input_ids = torch.randint(0, 1000, (1, 64))
    # Test prefill
    outputs = model(input_ids)
    print("Forward pass (prefill) successful!")
    
    # Test decode
    decode_input_ids = torch.randint(0, 1000, (1, 1))
    past_key_values = outputs.past_key_values
    outputs = model(decode_input_ids, past_key_values=past_key_values)
    print("Forward pass (decode) successful!")

if __name__ == "__main__":
    test_init()
