
import numpy as np

import torch
cuda_available = torch.cuda.is_available()
torch_version = torch.__version__


print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 打印PyTorch版本和CUDA可用性
print(f"PyTorch version: {torch_version}")
print(f"CUDA available: {cuda_available}")
cuda_version = torch.version.cuda
print(f"Current CUDA version: {cuda_version}")

import sys
print(sys.executable)

import torch
print(torch.version.cuda)
# 简单的测试GPU操作
x = torch.rand(5, 5).cuda()
y = torch.rand(5, 5).cuda()
print(x + y)  # 这应该在GPU上执行，而不会抛出错误