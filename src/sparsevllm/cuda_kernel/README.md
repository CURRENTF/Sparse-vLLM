# GPU Efficient Tensor Transfer (Zero-copy)

这个文件夹提取了 ShadowKV 中高效从 CPU (Pinned Memory) 读取 tensor 到 GPU 的 CUDA kernel 实现。

## 功能
- **Zero-copy Gather**: GPU kernel 直接从 CPU 内存读取数据，避免显式的全量 H2D 拷贝。
- **高效索引提取**: 支持带缓存逻辑的 gather 操作（D2D + H2D 混合优化）。
- **Python API**: 提供简单易用的 Python 接口。

## 目录结构
- `kernels/`: 包含 CUDA C++ 实现。
- `api.py`: Python 封装。
- `setup.py`: 编译脚本。
- `test_api_simple.py`: 基础功能测试脚本。

## 编译安装
在当前目录下运行：
```bash
pip install -e .
```

## 使用示例

### 简单 Gather
```python
import torch
from api import simple_gather

# CPU 数据必须是 pinned 才能保证高效
cpu_data = torch.randn(1, 1, 10000, 128, dtype=torch.bfloat16).pin_memory()
pos_ids = torch.tensor([[[0, 10, 500]]], device='cuda', dtype=torch.int32)

# 自动从 CPU 抓取数据到 GPU
gpu_result = simple_gather(cpu_data, pos_ids, map_size=3)
```

### 高级用法 (EfficientTransfer)
如果你需要频繁进行 gather 操作（如在 LLM 推理中），可以使用 `EfficientTransfer` 类来重用缓冲区。
```python
from api import EfficientTransfer
transfer = EfficientTransfer(batch_size=1, num_heads=1)
# ... 详见 api.py
```
