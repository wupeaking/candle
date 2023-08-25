# 使用torch进行校验结果是否正确

import numpy as np
from PIL import Image
from torchvision import models
import torch
import time
# from torchinfo import summary

m = models.resnet18(pretrained=True)
m.cuda(0)
m.eval()
resized_image = Image.open(
    "/DATA/candle/candle-examples/examples/resnet50/kitten.jpg").resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")  # shape(w, h, c)
img_data = np.transpose(img_data, (2, 0, 1))
img_data[:, :, :] = img_data[:, :, :] / 255.0  # 实际应该进行归一化 这里只做测试
img_data = np.expand_dims(img_data, axis=0)
ret = m(torch.tensor(img_data).cuda(0))
print(torch.softmax(ret, dim=1).max(dim=1))
lable_index = torch.softmax(ret, dim=1).argmax(dim=1)
print(lable_index)

# 测试一下时间

start_time = time.time()
i = 0
# m = m.cuda(0)
# torch.tensor(img_data).cuda(0)
while i < 10000:
    ret = m(torch.randn((1, 3, 224, 224)).cuda(0))
    print(ret.shape)
    i += 1
end_time = time.time() - start_time
elapsed_time = end_time/10000.0 * 1000
print(f"all time {end_time:.7f}s torch time: {elapsed_time:.7f} ms")


# 结果
# torch.return_types.max(
# values=tensor([0.3146], device='cuda:0', grad_fn=<MaxBackward0>),
# indices=tensor([282], device='cuda:0'))
# tensor([282], device='cuda:0')


def download_img():
    from PIL import Image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))

    # Preprocess the image and convert to tensor
    from torchvision import transforms
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)
    return img


def return_img_narray(img_path):
    # img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    # img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    # 重设大小为 224x224 resize(w, h)
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")  # shape(w, h, c)

    # ONNX 需要 NCHW 输入, 因此对数组进行转换
    img_data = np.transpose(img_data, (2, 0, 1))

    # 根据 ImageNet 进行标准化
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_stddev = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (
            img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

    # 添加 batch 维度
    img_data = np.expand_dims(norm_img_data, axis=0)
    return img_data
