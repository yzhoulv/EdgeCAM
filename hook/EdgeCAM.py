from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import os
from torchcam.methods import EdgeCAM
import torch.nn.functional as F
import torchvision
from mmengine.model import BaseModel
import torch
from models.metrics import ArcMarginProduct
import numpy as np


class EfficientB0(BaseModel):
    def __init__(self):
        super().__init__()
        self.efficient = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(1000, 512),
            torch.nn.Dropout(0.2)
        )
        self.metric = ArcMarginProduct(in_features=512, out_features=2)

    def forward(self, imgs, labels, mode):
        x = self.efficient(imgs)
        x = self.fc(x)
        x = self.metric(x, labels)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


@HOOKS.register_module()
class CalcEdgeCAMHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """

    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner):
        idx = runner.train_loop.epoch
        model_path = r'/data1/yangzhou/demo/CAMGuide/work_dir/epoch_' + str(idx) + '.pth'
        model = EfficientB0()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        print('index %d edge cam cal start!' % idx)
        self.calc_cam(model, idx)
        self.cam_merge(idx)
        print('index %d edge cam cal end!' % idx)
        del model

    def calc_cam(self, model, idx):
        trans = albu.Compose([
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            albu.Resize(512, 512, p=1),
            ToTensorV2()
        ])
        img_dir = r'/data1/yangzhou/datasets/cls/CASIAv2/Tp'
        model.eval()
        cam_extractor = EdgeCAM(model=model, target_layer='efficient.features.2')  #
        count = 0
        for img_name in os.listdir(img_dir):
            dst_path = os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0], str(idx) + '.jpg')
            image = cv2.imread(os.path.join(img_dir, img_name))
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input = trans(image=img)['image'].unsqueeze(0).cuda()
            input.requires_grad = True
            label = torch.LongTensor([1]).cuda()
            out, _ = model(input, label, mode='predict')
            out_s = F.softmax(out, dim=1)
            pred_class = out_s.squeeze(0).argmax().item()

            if not os.path.exists(os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0])):
                os.mkdir(os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0]))
            if pred_class == 0:
                cam_res = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.imwrite(dst_path, cam_res)
                count += 1
                continue
            cam_res = cam_extractor(1, out)
            cam_res = cam_res[0].cpu().numpy()
            cam_res = (np.squeeze(cam_res) ** 2) * 255.0
            cam_res = cam_res.astype(np.uint8)
            cam_res = cv2.resize(cam_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(dst_path, cam_res)
        print("%d images in training data wrong" % count)

    def cam_merge(self, idx):
        img_dir = r'/data1/yangzhou/datasets/cls/CASIAv2/Tp'
        for img_name in os.listdir(img_dir):
            dst_path = os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0], str(idx) + '.jpg')
            cam_total_path = os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0], 'total.jpg')
            latest_path = os.path.join(img_dir.replace('Tp', 'cam'), img_name.split('.')[0], 'latest.jpg')
            if os.path.exists(cam_total_path):
                cam_total = cv2.imread(cam_total_path, cv2.IMREAD_GRAYSCALE)
                cam_latest = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)
                img_new = np.dstack((cam_latest, cam_total))
                cam_new = np.max(img_new, axis=2)
                cv2.imwrite(cam_total_path, cam_new.astype(np.uint8))
                cv2.imwrite(latest_path, cam_latest)

            else:
                cam_total = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(cam_total_path, cam_total)
                cv2.imwrite(latest_path, cam_total)





