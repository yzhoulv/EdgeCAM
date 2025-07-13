import torch.nn.functional as F
import torchvision
from config.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner
from data.dataset import Dataset
from torch.utils import data
import torch
from torch.optim import SGD
from mmengine.optim import OptimWrapper
from models.metrics import ArcMarginProduct
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
from hook.EdgeCAM import CalcEdgeCAMHook


class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


class MMResNet34(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(1000, 512),
            torch.nn.Dropout(p=0.2),
        )
        self.metric = ArcMarginProduct(in_features=512, out_features=2, m=0.0)

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        x = self.fc(x)
        x = self.metric(x, labels)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


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


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


class Auc(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'label': gt.cpu().numpy(),
            'pred': score.cpu().numpy(),
        })

    def cal_auc(self, preds, label):
        preds = np.array(preds)
        label = np.array(label)
        # label[np.argmin(pred)] = 0
        pred = preds[:, 1]
        assert label.shape == pred.shape, print("cal auc error, shape not fit")
        auc_val = roc_auc_score(label, pred)
        # fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)
        # auc_val = auc(fpr, tpr)
        # plt.title('ROC')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.plot(fpr, tpr, '--*b', label="ours")
        # plt.legend()
        # plt.show()
        return auc_val

    def compute_metrics(self, results):
        labels = []
        preds = []
        for item in results:
            labels.extend(item['label'])
            preds.extend(item['pred'])
        auc_num = self.cal_auc(preds, labels)
        return dict(auc_score=auc_num)


class FScore(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'label': gt.cpu().numpy(),
            'pred': score.cpu().numpy(),
        })

    def cal_f1(self, preds, label):
        preds = np.array(preds)
        label = np.array(label)
        pred = np.argmax(preds, axis=1)
        assert label.shape == pred.shape, print("cal F1 error, shape not fit")

        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(label)):
            if label[i] == 1 and pred[i] == 1:
                TP += 1
            elif label[i] == 0 and pred[i] == 1:
                FP += 1
            elif label[i] == 1 and pred[i] == 0:
                FN += 1
            elif label[i] == 0 and pred[i] == 0:
                TN += 1

        sen = TP / (TP + FN)
        spe = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2.0 / (1.0 / precision + 1.0 / recall)
        acc = (TP + TN) / (TP + FP + FN + TN)
        print("specificity:%f, sensitivity:%f, acc:%f" % (spe, sen, acc))
        return F1

    def compute_metrics(self, results):
        labels = []
        preds = []
        for item in results:
            labels.extend(item['label'])
            preds.extend(item['pred'])
        fscore_num = self.cal_f1(preds, labels)
        return dict(f_score=fscore_num)


if __name__ == '__main__':
    # norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
    opt = Config()
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=opt.train_batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)

    val_dataset = Dataset(opt.val_root, opt.val_list, phase='val', input_shape=opt.input_shape)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=opt.test_batch_size,
                                     shuffle=False,
                                     num_workers=opt.num_workers)

    model = EfficientB0()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=10, convert_to_iter_based=False)
    default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, priority='LOW'))
    # custom_hooks = [dict(type='CalcEdgeCAMHook', interval=1, priority='VERY_LOW')]

    # for i in range(1, 100):
    runner = Runner(
        model=model,
        work_dir='./work_dir_col',
        train_dataloader=train_dataloader,
        optim_wrapper=OptimWrapper(optimizer), # dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9))
        train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=10),
        default_hooks=default_hooks,
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        val_evaluator=[dict(type=Accuracy), dict(type=FScore), dict(type=Auc)],
        # resume=True,
        param_scheduler=param_scheduler,
        # load_from='./work_dir_casia/epoch_' + str(i) + '.pth',
        launcher='pytorch',
        # custom_hooks=custom_hooks
    )
    runner.train()
