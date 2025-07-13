import os

import torch
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
from scipy import misc
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import os


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3, 512, 512)):
        self.phase = phase
        self.input_shape = input_shape
        self.root_path = root

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        self.imgs = [os.path.join(root, img[:-1]) for img in imgs]

        if self.phase == 'train':
            self.transforms = self._create_transformation(phase='train', img_size=input_shape[1])
        else:
            self.transforms = self._create_transformation(phase='val', img_size=input_shape[1])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.strip().split(' ')
        img_path = splits[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.phase == 'train' and img_path.find('Tp_') != -1:
            img_name = img_path.split('/')[-1].split('.')[0]
            cam_path = os.path.join(self.root_path, 'cam', img_name, 'latest.jpg')
            cam = self._create_mask(cam_path=cam_path, shape=(img.shape[0], img.shape[1]), p=0.8)
            try:
                img = img * np.repeat(cam[..., None], 3, axis=2)
            except:
                print(img_path)
        image = self.transforms(image=img.astype(np.uint8))['image']
        # augmented = self.transforms(image=image, mask=masks)
        # img, mask = augmented['image'], augmented['mask']
        label = np.long(splits[1])

        return image.float(), label

    def _create_mask(self, cam_path, shape, p=0.8):
        if os.path.exists(cam_path) and np.random.random_sample() > 1-p:
            cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)
            cam_np = np.array(cam, dtype=np.float32)
            cam_np = cam_np / 255.0
        else:
            cam_np = np.zeros(shape=shape, dtype=np.uint8)
        return 1.0 - cam_np

    # if os.path.exists(cam_path):
        #     cam = cv2.imread(cam_path, cv2.IMREAD_GRAYSCALE)
        #     cam_np = np.array(cam, dtype=np.float32)
        #     cam_np = cam_np / 255.0
        #     idxs = np.where(cam_np >= 0.3)
        #     sum_num = len(idxs[0])
        #     mask = np.random.choice(a=[0, 1], size=int(np.sum(sum_num)), replace=True, p=[p, 1-p])
        #     idxs = np.array(idxs)
        #     idxs = idxs[:, np.where(mask == 0)]
        #     idxs = np.squeeze(idxs, axis=1)
        #     cam_np[tuple(idxs)] = 0
        # else:
        #     cam_np = np.zeros(shape=shape, dtype=np.uint8)
        # return 1.0 - cam_np

    def __len__(self):
        return len(self.imgs)

    def _create_transformation(self, phase, img_size=512):
        transforms = albu.Compose([
            albu.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
            # albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # albu.RandomResizedCrop(img_size, img_size, scale=(0.50, 2.)),
            # albu.GaussNoise(p=0.5),
            # albu.Blur(blur_limit=10, p=0.2),
            albu.HorizontalFlip(p=0.5),
            # albu.RandomBrightnessContrast(p=0.5),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # albu.PadIfNeeded(min_height=512, min_width=512, p=1),
            albu.Resize(img_size, img_size, p=1),
            ToTensorV2()
        ])

        if phase == 'val':
            transforms = albu.Compose([
                albu.Resize(img_size, img_size, p=1),
                # albu.HorizontalFlip(p=0.5),
                albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

        return transforms


if __name__ == '__main__':
    dataset = Dataset(root='D:\\datasets\\Manipulation\\cls\\casiav2\\',
                      data_list_file='D:\\datasets\\Manipulation\\cls\\casiav2\\casiav2.txt',
                      phase='val',
                      input_shape=(3, 512, 512))

    trainloader = data.DataLoader(dataset, batch_size=4)
    for i, (data, label) in enumerate(trainloader):
        data = data[:, 0:3, :, :]
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        print(img.shape)
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        img *= np.array([0.229, 0.224, 0.225])
        img += np.array([0.485, 0.456, 0.406])
        img *= np.array([255, 255, 255])
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)