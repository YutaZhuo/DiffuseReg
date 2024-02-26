from torch.utils.data import Dataset
import data.util_3D as Util
import os
import numpy as np
import torch
import scipy.io as sio
import json
import SimpleITK as sitk


class ACDCDataset(Dataset):
    def __init__(self, dataroot, fineSize, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split + '.json')
        with open(datapath, 'r') as f:
            self.imageNum = json.load(f)
            print(self.imageNum)
        for it in self.imageNum:
            it['image_ED'] = os.path.join(dataroot, it['image_ED'])
            it['image_ES'] = os.path.join(dataroot, it['image_ES'])
            it['label_ED'] = os.path.join(dataroot, it['label_ED'])
            it['label_ES'] = os.path.join(dataroot, it['label_ES'])

        self.data_len = len(self.imageNum)
        self.fineSize = fineSize

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # ED-fixed, ES-moving
        dataPath = self.imageNum[index]
        # data_ = sio.loadmat(dataPath)
        dataA = dataPath['image_ED']
        dataA = sitk.ReadImage(dataA)
        dataA = sitk.GetArrayFromImage(dataA).astype(np.float32)
        # print(dataA.shape)
        dataB = dataPath['image_ES']
        dataB = sitk.ReadImage(dataB)
        dataB = sitk.GetArrayFromImage(dataB).astype(np.float32)
        label_dataA = dataPath['label_ED']
        label_dataA = sitk.ReadImage(label_dataA)
        label_dataA = sitk.GetArrayFromImage(label_dataA)
        label_dataB = dataPath['label_ES']
        label_dataB = sitk.ReadImage(label_dataB)
        label_dataB = sitk.GetArrayFromImage(label_dataB)

        # data normalize, Step1: value range[0, 1]
        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nd, nh, nw = dataA.shape
        # print(dataA.shape,dataB.shape)

        sh = int((nh - self.fineSize[1]) / 2)
        sw = int((nw - self.fineSize[2]) / 2)
        dataA = dataA[:, sh:sh + self.fineSize[1], sw:sw + self.fineSize[2]]
        dataB = dataB[:, sh:sh + self.fineSize[1], sw:sw + self.fineSize[2]]
        label_dataA = label_dataA[:, sh:sh + self.fineSize[1], sw:sw + self.fineSize[2]]
        label_dataB = label_dataB[:, sh:sh + self.fineSize[1], sw:sw + self.fineSize[2]]

        if nd >= self.fineSize[0]:
            sd = int((nd - self.fineSize[0]) / 2)
            dataA = dataA[sd:sd + self.fineSize[0]]
            dataB = dataB[sd:sd + self.fineSize[0]]
            label_dataA = label_dataA[sd:sd + self.fineSize[0]]
            label_dataB = label_dataB[sd:sd + self.fineSize[0]]
        else:
            sd = int((self.fineSize[0] - nd) / 2)
            dataA_ = np.zeros(self.fineSize)
            dataB_ = np.zeros(self.fineSize)
            dataA_[sd:sd + nd] = dataA
            dataB_[sd:sd + nd] = dataB
            label_dataA_ = np.zeros(self.fineSize)
            label_dataB_ = np.zeros(self.fineSize)
            label_dataA_[sd:sd + nd] = label_dataA
            label_dataB_[sd:sd + nd] = label_dataB
            dataA, dataB = dataA_, dataB_
            label_dataA, label_dataB = label_dataA_, label_dataB_

        # data normalize, Step2: value range[-1, 1]
        [fixed, moving] = Util.transform_augment([dataA, dataB],
                                                 split=self.split,
                                                 min_max=(-1, 1))

        fixedM = torch.from_numpy(label_dataA).float().unsqueeze(0)
        movingM = torch.from_numpy(label_dataB).float().unsqueeze(0)

        return {'M': moving, 'F': fixed, 'MS': movingM, 'FS': fixedM, 'Index': index}
