import cv2
import numpy as np
import os

def cal_fscore(pred, label):
    # TP    predict 和 label 同时为1
    TP = ((pred == 1) & (label == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((pred == 0) & (label == 0)).sum()
    # FN    predict 0 label 1
    FN = ((pred == 0) & (label == 1)).sum()
    # FP    predict 1 label 0
    FP = ((pred == 1) & (label == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2. * r * p / (r + p)

    iou = TP / ((pred == 1).sum() + (label == 1).sum() - TP)

    if np.isnan(F1):
        F1 = 0
    if np.isnan(iou):
        iou = 0

    return F1


def erode_demo(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dst = cv2.erode(image, kernel, 2)
    dilate = cv2.dilate(dst, kernel, 2)
    # cv2.imshow("erode", binary)
    return dilate
    

def merge_ucm():
    ucm_dir = r'D:\datasets\Manipulation\seg\Columbia\ucm'
    img_dir = r'D:\datasets\Manipulation\seg\Columbia\image'
    # cam_dir = r'D:\datasets\Manipulation\seg\Columbia\cam_total'
    mask_dir = r'D:\datasets\Manipulation\seg\Columbia\view'
    cam_img_dir = r'D:\datasets\Manipulation\seg\Columbia\cam'
    count = 0
    fscore = 0

    for ucm in os.listdir(ucm_dir):
        ucm_img = cv2.imread(os.path.join(ucm_dir, ucm), cv2.IMREAD_GRAYSCALE)
        ucm_img[ucm_img < 230] = 0
        ucm_img[ucm_img >= 230] = 255
        ucm_img = cv2.medianBlur(ucm_img, 5)
        # ucm_img = erode_demo(ucm_img)
        # ucm_img[ucm_img < 100] = 0
        # ucm_img[ucm_img >= 100] = 255
        ucm_img = 255 - ucm_img
        ucm_img = np.array(ucm_img, dtype=np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ucm_img, connectivity=8)

        output = np.zeros((ucm_img.shape[0], ucm_img.shape[1], 3), np.uint8)
        for i in range(1, num_labels):
            mask = labels == i
            output[:, :, 0][mask] = np.random.randint(0, 255)
            output[:, :, 1][mask] = np.random.randint(0, 255)
            output[:, :, 2][mask] = np.random.randint(0, 255)
        

        votes = np.zeros(num_labels)
        
        if os.path.exists(os.path.join(cam_img_dir, ucm.replace('.jpg', ''))):
            
            for cam in os.listdir(os.path.join(cam_img_dir, ucm.replace('.jpg', ''))):
                flags = np.zeros(num_labels)
                cam_img = cv2.imread(os.path.join(cam_img_dir, ucm.replace('.jpg', ''), cam), cv2.IMREAD_GRAYSCALE)
                cam_img = cv2.resize(cam_img, (ucm_img.shape[1], ucm_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                res = np.argwhere(cam_img >= 245)
                for r in res: 
                    idx = labels[r[0]][r[1]]
                    if flags[idx-1] == 0:
                        votes[idx-1] += 1
                        flags[idx-1] = 1
            
            votes = votes[0 : -2]
            if len(votes) == 0:
                continue
            max_num = votes.max()
            # max_idx = np.argwhere(max_num == votes)
            max_idxs = np.argwhere(votes == max_num)
            for max_idx in max_idxs:
                labels[labels == max_idx+1] = 255
            
            labels[labels != 255] = 0
            labels = np.array(labels, dtype=np.uint8)

            print(votes)
            cv2.imshow('original', output)
            cv2.imshow('merge res', np.array(labels, dtype=np.uint8))
            cv2.waitKey(0)
            

            mask = cv2.imread(os.path.join(mask_dir, ucm.replace('.jpg', '_gt.png')), cv2.IMREAD_GRAYSCALE)
            mask[mask==255] = 1
            labels[labels==255] = 1
            labels = np.array(labels, dtype=np.uint8)
            labels =  cv2.resize(labels, (mask.shape[1], mask.shape[0]))

            mask = np.array(mask, dtype=np.uint8)
            labels = np.array(labels, dtype=np.uint8)
            tmp_score = cal_fscore(mask, labels)
            print('img:%s, f1:%f' %(ucm, tmp_score))
            fscore += tmp_score
            count += 1
    
    print('count:%d, res:%f' % (count, fscore/count))


if __name__ == '__main__':
    merge_ucm()



   # # mask_dir = r'D:\datasets\Manipulation\seg\Columbia\mask'
#    for m in os.listdir(mask_dir):
#         mask = cv2.imread(os.path.join(mask_dir, m), cv2.IMREAD_GRAYSCALE)
#         view = mask * 255
#         cv2.imwrite(os.path.join(mask_dir.replace('mask', 'view'), m), view)
