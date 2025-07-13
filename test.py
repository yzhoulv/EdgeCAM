import torch
from config.config import Config
from data.dataset import Dataset
from models.resnet import get_convnext_model
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from cam.scorecam import *
import torchvision
import os



def cal_f1(pres, label):
    pred = np.argmax(pres, axis=1)
    pred = np.array(pred)
    label = np.array(label)
    assert label.shape == pred.shape, print("cal F1 error, shape not fit")

    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1:
            TP += 1
        if label[i] == 0 and pred[i] == 1:
            FP += 1
        if label[i] == 1 and pred[i] == 0:
            FN += 1
        if label[i] == 0 and pred[i] == 0:
            TN += 1    

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)  
    F1 = 2 / (1/precision + 1/recall)
    acc = (TP+TN) / (TP+FP+FN+TN)   
    return precision, recall, F1, acc


def cal_auc(preds, label):
    preds = np.array(preds)
    label = np.array(label)
    # label[np.argmin(pred)] = 0
    pred = preds[:, 1]
    assert label.shape == pred.shape, print("cal auc error, shape not fit")
    # auc_val = roc_auc_score(label, pred)
    fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    # plt.title('ROC')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.plot(fpr, tpr, '--*b', label="ours")
    # plt.legend()
    # plt.show()
    return auc_val


def inference(net, test_root, test_list):
    opt = Config()
    net.eval()
    test_dataset = Dataset(test_root, test_list, phase='val', input_shape=opt.input_shape)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    outputs = []
    labels = []
    for i, data in enumerate(test_dataloader):
        inputs, target = data
        inputs = inputs.cuda()
        # target = target.cuda()
        with torch.no_grad():
            feats = net(inputs)
        feats = F.softmax(feats, dim=1)
        output = feats.data.cpu().numpy()
        outputs.extend(output)
        labels.extend(target)

    #     _, predict = torch.max(output, 1)
    #     # print(predict, " target: ", target)
    #     for j in range(len(predict)):
    #         if predict[j] == target[j]:
    #             num += 1
    #     # print("i: ", i, " num: ", num)
    # # print('acc: ', num / len(test_data))
    # test_acc = num / len(test_dataset)
    return outputs, labels

def test(net, test_root, test_list):
    preds, labels = inference(net, test_root, test_list)
    auc_val = cal_auc(preds, labels)
    precision, recall, F1, acc = cal_f1(preds, labels)
    print("precision: %f, recall: %f, F1: %f, acc: %f, auc: %f" % (precision, recall, F1, acc, auc_val))
    return acc


def print_model():
    model = torchvision.models.resnet18()
    for name, param in model.named_parameters():
        print(name)
    for idx in model._modules.keys():
        print(idx, '->', model._modules[idx])


    
def calc_cam(model, idx):
    # for name, param in model.named_parameters():
    #     print(name)
    
    # for idx in model._modules.keys():
    #     print(idx, '->', model._modules[idx])

    
    trans = albu.Compose([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    img_dir = r'D:\datasets\Manipulation\seg\casia\CASIA_V2\image'
    for img_name in os.listdir(img_dir):
        dst_path = os.path.join(img_dir.replace('image', 'cam'), img_name.split('.')[0], str(idx)+'.jpg')
        if os.path.exists(dst_path):
            continue
        img = cv2.imread(os.path.join(img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input = trans(image=img)['image'].unsqueeze(0).cuda()
        input.requires_grad = True
        model_dict = dict(type='convnext_tiny', arch=model, layer_name='features', input_size=(3, 256, 384))
        scorecam = ScoreCAM(model_dict)
        cam_res, pred_class = scorecam(input=input)
        if pred_class == 1:
            cam_res = cam_res.cpu().numpy()
            cam_res = np.squeeze(cam_res)*255
            cam_res = cam_res.astype(np.uint8)
            if not os.path.exists(os.path.join(img_dir.replace('image', 'cam'), img_name.split('.')[0])):
                os.mkdir(os.path.join(img_dir.replace('image', 'cam'), img_name.split('.')[0]))
            cv2.imwrite(dst_path, cam_res)
            print(dst_path)
        
        
        # cam_res = cv2.cvtColor(cam_res, cv2.COLOR_GRAY2RGB)
        # heat_img = cv2.applyColorMap(cam_res, cv2.COLORMAP_HSV) # 注意此处的三通道热力图是cv2专有的GBR排列
        # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像
        # img_add = cv2.addWeighted(img, 0.3, heat_img, 0.7, 0)
        # cv2.imshow(str(i), img_add)
        # i += 1
        # cv2.waitKey(0)
    # with torch.no_grad():
    #     feats = model(input)
    #     feats = F.softmax(feats, dim=1)
    #     print(feats)



if __name__ == '__main__':
    # print_model()
    for i in range(1, 100):
        model_path = r'./output/convnext_tiny_new_' + str(i) + '.pth'
        print(model_path)
        model = get_convnext_model()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        calc_cam(model=model, idx=i)
        model.cuda()
    # model.cuda()
    # preds, labels = inference(net=model, test_root='D:\\datasets\\Manipulation\\cls\\COVERAGE\\', test_list='D:\\datasets\\Manipulation\\cls\\COVERAGE\\COVERAGE.txt')
    # auc = 0 # cal_auc(preds, labels)
    # precision, recall, F1, acc = cal_f1(preds, labels)
    # print("precision: %f, recall: %f, F1: %f, acc: %f, auc: %f" % (precision, recall, F1, acc, auc))


    