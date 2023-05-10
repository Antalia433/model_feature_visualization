import sys

sys.path.append('./')
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
import pandas as pd
import os


class MyDataset(data.Dataset):

    def __init__(self, root_dir='./', img_dir='./', imgsize=448, new_crop=False):
        self.Rootdir = root_dir
        info_file = pd.read_csv(img_dir)
        self.ImgList = info_file['name']
        self.ImgLabel = info_file['label']
        self.new_crop = new_crop
        self.img_size = imgsize
        self.MEAN = (0.485, 0.456, 0.406)
        self.STD = (0.229, 0.224, 0.225)

    def for_test(self, image, corp):
        TRANS_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(corp),
            # transforms.RandomCrop(self.img_size), # banch_size设为1就不用把img裁成统一size、实测test用整张图，acc会高一点
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])
        return TRANS_test(image)

    def __len__(self):
        return len(self.ImgList)

    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.Rootdir, self.ImgList[index])).convert('RGB')
        except:
            index = index + 1
            img = Image.open(os.path.join(self.Rootdir, self.ImgList[index])).convert('RGB')
        crop = (self.img_size, self.img_size)
        # 做等比例缩放、保证短边长为img.size
        if self.new_crop:
            w, h = img.size
            # number = random.randint(self.img_size, self.img_size + 56)
            number = self.img_size  # + number  # 按论文来做的、但加了这个随机数之后acc会一直变动，评估的时候有点噩梦
            if w == h:
                pass
            elif w > h:
                # corp的顺序是高、宽
                crop = (number, int(number / h * w))
            else:
                crop = (int(number / w * h), number)

        img = self.for_test(img, crop)
        label = self.ImgLabel[index]

        return img, label, os.path.join(self.Rootdir, self.ImgList[index])


def main(root_path, info_path):
    test_set = MyDataset(root_dir=root_path,
                         img_dir=info_path,
                         imgsize=448, new_crop=True)
    test_loader = data.DataLoader(dataset=test_set,
                                  batch_size=1,
                                  # num_workers=1,
                                  # drop_last=True,
                                  pin_memory=True,
                                  )
    return test_loader

