from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        """
        :param images_path: 图像文件路径列表
        :param images_class: 对应的类别标签列表
        :param transform: 应用于所有图像的转换操作
        """
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    # 只针对RGB的所有数据
    # def __getitem__(self, item):
    #     img = Image.open(self.images_path[item])
    #     # RGB为彩色图片，L为灰度图片
    #     if img.mode != 'RGB':
    #         raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
    #     label = self.images_class[item]

    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, label

    # 将灰度图也转化为RGB格式
    # def __getitem__(self, item):
    #     img = Image.open(self.images_path[item]).convert('RGB')  # 确保所有图像都转换为RGB模式
    #     label = self.images_class[item]

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     return img, label

    # 添加数据增强后的

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert(
            'RGB')  # 确保所有图像都转换为RGB模式
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
