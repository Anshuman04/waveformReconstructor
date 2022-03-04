import os
import copy
import glob
import torch
import random
import logging
import numpy as np
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    def __init__(self, base_dir, gap_len=30, augment=False, dataType="test"):
        self.dataType = dataType
        self.allData = self.loader(base_dir, gap_len=gap_len, augment=augment)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.allData)

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.allData[index]["sample"]
        y = self.allData[index]

        return X.reshape(1,1,-1), y

    def gap_generator(self, dat, gap_len=30):
        '''
        Build a batch of waveforms with gaps
        data:    numpy array with raw data
        bs:      batch size to be generated
        win_len: window length of each example in seconds
        gap_len: gap length in seconds  # Hardcoded for x points
        fs:      sample rate of signal in hertz
        '''
        copyData = copy.copy(dat)
        sampleDict = {}
        gapStartIdx = np.random.randint(len(dat) - gap_len)
        gapEndIdx = gapStartIdx + gap_len
        cutData = dat[gapStartIdx: gapEndIdx]
        copyData[gapStartIdx: gapEndIdx] = 0
        sampleDict["gapStartIdx"] = gapStartIdx
        sampleDict["gapEndIdx"] = gapEndIdx
        sampleDict["original"] = dat
        sampleDict["sample"] = copyData
        return sampleDict

    def loader(self, baseDir, gap_len=30, augment=False):
        assert os.path.exists(baseDir), "Base Directory [{}] does not exist".format(baseDir)
        dataFileRegex = "{}*.npy".format(self.dataType)      # Hardcoded
        fileList = glob.glob(os.path.join(baseDir, dataFileRegex))
        samplePoints = []
        for fileName in fileList:
            logging.info("Reading datafile: {}".format(fileName))
            fileData = np.load(fileName, allow_pickle=True).astype("float32")
            for item in fileData:
                if augment:
                    itemRev = item[::-1]
                    samplePoints.append(self.gap_generator(itemRev, gap_len))
                samplePoints.append(self.gap_generator(item, gap_len))
        random.shuffle(samplePoints)
        return samplePoints
