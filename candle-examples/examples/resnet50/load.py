# coding=utf-8
from torchvision import models
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from torchinfo import summary

if __name__ == "__main__":
    # 下载模型
    m = models.resnet18(pretrained=True)
    # 保存权重
    m.eval()
    torch.save(m.state_dict(), 'resnet18_weight.pth')
    # 打印torch模型
    summary(m, (1, 3, 224, 224))

    # 导出到onnx
    torch.onnx.export(m, torch.randn((1, 3, 224, 224)),
                      "resnet18.onnx", verbose=False)
    # 转换到safetensor
    save_file(m.state_dict(), "resnet18.safetensors")
    with safe_open("resnet18.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            print(key, f.get_tensor(key))
