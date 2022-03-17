import torch
import copy
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from customDataset import SeismicDataset
from torch.utils.data import DataLoader
from defaultConfig import get_config
from model import WaveformReconstructor

TEST_MODEL = "trainedModels\\trainedModel_epoch20.pt"
DATAPATH = r"C:\Users\anshu\JupyterNotebooks\Seismic-Inpainting\dataset"
OP_DIR = "mseGapOnlyRes"
BATCH_SIZE = 1
NUM_WORKERS = 1

def loadArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    allArgs = parser.parse_args()
    return allArgs

def compiler(img,label,op,idx=0):
    l = {}
    op = op.detach()
    op = op.cpu()
    l["img"] = img
    l["label"]= label
    l["op"] = op
    orig = label['original'][idx]
    recons = op[idx]
    recons_val = recons[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]]
    orig_val = orig[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]]
    l1norm = np.linalg.norm(orig_val - recons_val, ord=1)
    l["loss"] = l1norm
    return l

def plotter(img, label, op, bNum, idx=0, keyName="result"):
    orig = label['original'][idx]
    gap = img[idx][0][0]
    recons = op[idx]
    final = copy.deepcopy(orig)
    final[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]] = recons[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]]
    # final = final.detach()
    pStart = label['gapStartIdx'][idx] - 400
    pEnd = label['gapEndIdx'][idx] + 400
    fig, ax = plt.subplots(3)
    ax[0].plot(orig[pStart:pEnd])
    ax[0].axvline(x = 400, color = 'r')
    ax[0].axvline(x = 460, color = 'r')
    ax[1].plot(gap.cpu()[pStart:pEnd])
    ax[1].axvline(x = 400, color = 'r')
    ax[1].axvline(x = 460, color = 'r')
    ax[2].plot(final[pStart:pEnd])
    ax[2].axvline(x = 400, color = 'r')
    ax[2].axvline(x = 460, color = 'r')
    plt.savefig(os.path.join(OP_DIR, "{}_{}_{}.png".format(keyName, bNum, idx)))
    fig.clear()

if __name__ == "__main__":
    storeList = []
    testContainer = SeismicDataset(DATAPATH, gap_len=60, augment=False, dataType="test")
    testGenerator = DataLoader(testContainer,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=NUM_WORKERS)
    allArgs = loadArguments()
    config = get_config(allArgs)
    model = WaveformReconstructor(config, img_size=allArgs.img_size, num_classes=allArgs.num_classes).cuda()
    model.load_state_dict(torch.load(TEST_MODEL))
    model.eval()
    import pdb
    pdb.set_trace()
    for batchIdx, batchData in enumerate(testGenerator):
        img, label = batchData
        img = img.cuda()
        img = img.float()
        op = model(img)
        storeList.append(compiler(img, label, op))

    import pdb
    pdb.set_trace()
    storeList.sort(key=lambda x: x["loss"])
    n = len(storeList)
    # for i in range(6):
    #     plotter(storeList[i]["img"], storeList[i]["label"], storeList[i]["op"], bNum=i + 1, keyName="best")
    #     plotter(storeList[n-i-1]["img"], storeList[n-i-1]["label"], storeList[n-i-1]["op"], bNum=i + 1, keyName="worst")

    # dump all output ~ 152MB on disk
    for i in range(len(storeList)):
        plotter(storeList[i]["img"], storeList[i]["label"], storeList[i]["op"], bNum=i + 1, keyName="result")








