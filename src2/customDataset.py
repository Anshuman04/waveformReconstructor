import os
import copy
import glob
import torch
import random
import logging
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import lfilter, butter, decimate

class SeismicDataset(Dataset):
    def __init__(self, base_dir, gap_len=60, augment=False, dataType="test"):
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

    @property
    def params(self):
        return {
            "flo": .3,
            "fhi": 5,
            "wlen": 30,
            "glen": 1,
            "fsorig": 40,
            "fs": 80
        }

    def gap_generator(self, dat, gap_len=30):
        '''
        Build a batch of waveforms with gaps
        data:    numpy array with raw data
        bs:      batch size to be generated
        win_len: window length of each example in seconds
        gap_len: gap length in seconds  # Hardcoded for x points
        fs:      sample rate of signal in hertz
        '''
        copyData = copy.deepcopy(dat)
        sampleDict = {}
        gapStartIdx = np.random.randint(len(dat) - gap_len)
        gapEndIdx = gapStartIdx + gap_len
        cutData = dat[gapStartIdx: gapEndIdx]
        copyData[gapStartIdx: gapEndIdx] = 0
        sampleDict["gapStartIdx"] = gapStartIdx
        sampleDict["gapEndIdx"] = gapEndIdx
        sampleDict["original"] = torch.from_numpy(dat.copy())
        sampleDict["sample"] = torch.from_numpy(copyData.copy())
        return sampleDict

    def loader(self, baseDir, gap_len=60, augment=False):
        assert os.path.exists(baseDir), "Base Directory [{}] does not exist".format(baseDir)
        dataFileRegex = "{}*.npy".format(self.dataType)      # Hardcoded
        fileList = glob.glob(os.path.join(baseDir, dataFileRegex))
        samplePoints = []
        for fileName in fileList:
            logging.info("Reading datafile: {}".format(fileName))
            fileData = np.load(fileName, allow_pickle=True).astype("float32")
            for item in fileData:
                dat1 = self.butter_bandpass_filter(item, self.params["flo"],
                                                   self.params["fhi"], fs=self.params["fsorig"])
                upsample = self.upsample(dat1, self.params["fsorig"], self.params["fs"],self.params["wlen"])
                if augment:
                    itemRev = upsample[::-1]
                    samplePoints.append(self.gap_generator(itemRev, gap_len))
                samplePoints.append(self.gap_generator(upsample, gap_len))
        random.shuffle(samplePoints)
        # import pdb
        # pdb.set_trace()
        return samplePoints

    def butter_bandpass(self, lowcut, highcut, fs, order=8):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=8):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return lfilter(b, a, data)

    def upsample(self, x, fs_old, fs_new, win_len):
        win_len = len(x)/fs_old
        t_old = np.linspace(0, int(win_len-1/fs_old), int(win_len*fs_old))
        t_new = np.linspace(0, int(win_len-1/fs_new), int(win_len*fs_new))
        x = np.interp(t_new, t_old, x)
        return x
