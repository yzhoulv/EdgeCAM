import torch
from models.resnet import get_convnext_model
import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torchvision
import os
from torchcam.methods import SmoothGradCAMpp
import torch.nn.functional as F


def print_model():
    model = torchvision.models.resnet18()
    for name, param in model.named_parameters():
        print(name)
    for idx in model._modules.keys():
        print(idx, '->', model._modules[idx])


def calc_cam(model, idx):
    trans = albu.Compose([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    img_dir = r'D:\datasets\Manipulation\seg\casia\CASIA_V1\image'
    model.eval()
    cam_extractor = SmoothGradCAMpp(model)

    for img_name in os.listdir(img_dir):
        # img_name = r'D:\datasets\Manipulation\seg\casia\CASIA_V2\image\Tp_D_CND_S_N_txt00028_txt00006_10848.jpg'
        dst_path = os.path.join(img_dir.replace('image', 'cam_ca'), img_name.split('.')[0], str(idx) + '.jpg')
        # if os.path.exists(dst_path):
        # continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        # img = cv2.resize(img_ori, dsize=(0, 0), fx=0.25, fy=0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = trans(image=img)['image'].unsqueeze(0).cuda()
        input.requires_grad = True
        # with torch.no_grad():
        out = model(input)
        out_s = F.softmax(out)
        pred_class = out_s.squeeze(0).argmax().item()
        if pred_class == 0 or out_s[0][1] < 0.8:
            continue
        cam_res = cam_extractor(1, out)
        cam_res = cam_res[0].cpu().numpy()
        # print(cam_res)
        # cam_res[cam_res < 0.9] = 0
        cam_res = np.squeeze(cam_res) * 255
        cam_res = cam_res.astype(np.uint8)
        if not os.path.exists(os.path.join(img_dir.replace('image', 'cam_ca'), img_name.split('.')[0])):
            os.mkdir(os.path.join(img_dir.replace('image', 'cam_ca'), img_name.split('.')[0]))
        # cv2.imshow("cam_res", cam_res)
        cam_res = cv2.resize(cam_res, (img.shape[1], img.shape[0]))
        # cv2.imwrite(dst_path, cam_res)
        print(dst_path)
        cam_res = cv2.cvtColor(cam_res, cv2.COLOR_GRAY2RGB)
        heat_img = cv2.applyColorMap(cam_res, cv2.COLORMAP_HSV)  # 注意此处的三通道热力图是cv2专有的GBR排列
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
        img_add = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)
        cv2.imshow('cam', img_add)
        cv2.waitKey(0)
        # break


if __name__ == '__main__':
    # print_model()
    for i in range(99, 100):
        model_path = r'./output_casia_v2/convnext_tiny_new_' + str(i) + '.pth'
        print(model_path)
        model = get_convnext_model(pretrain=False)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        calc_cam(model=model, idx=i)
