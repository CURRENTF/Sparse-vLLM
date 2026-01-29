import torch
import torch.nn as nn
import time
import fire


@torch.no_grad()
def test_overhead(batch_sizes=2048, use_compile=True, scenario="small", in_features=None, mid_features=None, out_features=None):
    """
    测试 Tensor 运算流程的开销
    
    Args:
        batch_sizes: 输入 tensor 的行数
        use_compile: 是否测试 torch.compile 后的性能
        scenario: 'small' (默认测试) 或 'llm' (模拟 Llama-3-8B MLP)
        in_features: 自定义输入维度 (覆盖 scenario)
        mid_features: 自定义中间维度 (覆盖 scenario)
        out_features: 自定义输出维度 (覆盖 scenario)
    """
    # 检查 GPU 是否可用，优先使用 GPU 进行测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设定数据类型为 bfloat16
    dtype = torch.bfloat16
    print(f"使用数据类型: {dtype}")

    # 确定维度配置
    if in_features is not None and mid_features is not None and out_features is not None:
        pass # 使用用户自定义
    elif scenario == "llm":
        # Llama-3-8B MLP 规模 (大约)
        # Hidden=4096, Intermediate=14336
        in_features = 4096
        mid_features = 14336
        out_features = 4096
        print(f"场景: Llama-3-8B MLP 模拟")
    else:
        # 原默认配置
        in_features = 256
        mid_features = 4096
        out_features = 1024
        print(f"场景: Small Model (Default)")

    print(f"维度配置: In={in_features}, Mid={mid_features}, Out={out_features}")

    # 处理 batch_sizes 参数
    if isinstance(batch_sizes, int):
        batch_sizes_list = [batch_sizes]
    elif isinstance(batch_sizes, str):
        # 处理可能的字符串输入，如 "1,2,3"
        if ',' in batch_sizes:
            batch_sizes_list = [int(x.strip()) for x in batch_sizes.split(',')]
        else:
            batch_sizes_list = [int(batch_sizes)]
    elif isinstance(batch_sizes, (list, tuple)):
        batch_sizes_list = batch_sizes
    else:
        batch_sizes_list = [batch_sizes]

    print(f"待测试 Batch Sizes: {batch_sizes_list}")

    # 定义维度
    # in_features, mid_features, out_features 已在上方定义

    for batch_size in batch_sizes_list:
        print(f"\n{'='*20} Batch Size: {batch_size} {'='*20}")
        
        # 初始化输入 tensor (batch_size x 256) 并转换为 bf16
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)

        # 定义操作流程并转换为 bf16
        # 每次循环重新创建模型，确保环境隔离
        model = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.GELU(),
            nn.Linear(mid_features, out_features)
        ).to(device).to(dtype)

        def run_benchmark(target_model, label):
            print(f"\n--- 开始测试: {label} (BS={batch_size}) ---")
            
            # 预热 (Warmup)，torch.compile 需要在预热阶段完成编译
            warmup_iters = 50 if "Compile" in label else 20
            # print(f"正在进行预热 ({warmup_iters} 次迭代)...")
            for _ in range(warmup_iters):
                _ = target_model(x)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()

            # 开始正式测试
            iters = 1000
            # print(f"执行 {iters} 次迭代中...")
            
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                for _ in range(iters):
                    _ = target_model(x)
                end_event.record()
                
                torch.cuda.synchronize()
                total_time_ms = start_event.elapsed_time(end_event)
                avg_time_ms = total_time_ms / iters
            else:
                # CPU 测量方式
                start_time = time.perf_counter()
                for _ in range(iters):
                    _ = target_model(x)
                end_time = time.perf_counter()
                avg_time_ms = (end_time - start_time) / iters * 1000

            print(f"单次流程平均耗时: {avg_time_ms:.4f} ms")
            print(f"每秒可执行次数: {1000 / avg_time_ms:.2f} iterations/s")
            return avg_time_ms

        # 1. 测试 Eager 模式
        eager_time = run_benchmark(model, "Eager Mode")

        # 2. 测试 Torch Compile 模式
        if use_compile:
            if hasattr(torch, 'compile'):
                print(f"\n正在编译模型 (torch.compile) - BS={batch_size}...")
                compiled_model = torch.compile(model)
                compile_time = run_benchmark(compiled_model, "Torch Compile")
                print(f"\n--- 对比总结 (BS={batch_size}) ---")
                print(f"Eager 耗时: {eager_time:.4f} ms")
                print(f"Compile 耗时: {compile_time:.4f} ms")
                print(f"加速比: {eager_time / compile_time:.2f}x")
            else:
                print("\nCurrent PyTorch 版本不支持 torch.compile")


if __name__ == "__main__":
    # 使用 fire 暴露命令行参数
    fire.Fire(test_overhead)
