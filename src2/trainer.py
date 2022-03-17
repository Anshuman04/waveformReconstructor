import os
import sys
import time
import logging
import argparse
import torch
import numpy as np

from tqdm import tqdm
import robust_loss_pytorch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss

from customDataset import SeismicDataset
from defaultConfig import get_config
from model import WaveformReconstructor
import torch.nn as nn


BATCH_SIZE = 4
nn.CrossEntropyLoss
NUM_WORKERS = 2
DATAPATH = r"C:\Users\anshu\JupyterNotebooks\Seismic-Inpainting\dataset"
assert os.path.exists(DATAPATH), "Dataset path invalid: {}".format(DATAPATH)

def setupLogger():
    """
        Logger setup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", '%m-%d-%Y %H:%M:%S')
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    stdout.setFormatter(formatter)
    logger.addHandler(stdout)
    logging.debug("Setting up logger completed")

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
    parser.add_argument('--lossName', default="mse", choices=["mse", "adaptive"], help='LossName')
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


def plotASample(img, label, op, idx, scaleFactor=1.0):
    import copy
    from matplotlib import pyplot as plt
    orig = label['original'][idx]
    gap = img[idx][0][0]
    recons = op[idx]

    # bumpUpTest = min(recons)
    # if bumpUpTest < 0:
    #     recons = recons + abs(bumpUpTest)
    # bumpUpRef = min(orig)
    # tempRef = copy.deepcopy(orig)
    # if bumpUpRef < 0:
    #     tempRef = tempRef + abs(bumpUpRef)
    # scaleFactor = tempRef.mean() / recons.mean()
    #
    # recons = recons - abs(bumpUpTest)
    recons = recons * float(scaleFactor)

    final = copy.deepcopy(orig)
    final[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]] = recons[label['gapStartIdx'][idx] : label['gapEndIdx'][idx]]
    final = final.detach()

    pStart = label['gapStartIdx'][idx] - 400
    pEnd = label['gapEndIdx'][idx] + 400
    fig, ax = plt.subplots(3)
    ax[0].plot(orig[pStart:pEnd])
    ax[1].plot(gap.cpu()[pStart:pEnd])
    ax[2].plot(final.cpu()[pStart:pEnd])
    plt.savefig("result{}.png".format(idx))

def getLoss(lossName="mse"):
    assert lossName in ["mse", "adaptive"]
    if lossName == "mse":
        return MSELoss()
    if lossName == "adaptive":
        return robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = 14400,
                                                                          float_dtype = np.float32,
                                                                          device=0)
    raise Exception("INVALID SOFTWARE FLOW")

def computeLoss(lossObj, test, ref):
    """
    Biased loss 0.67 Normal + 0.33 biased for gap boundaries
    """
    # normalLoss = lossObj(test, ref['original'].cuda().float())
    # B, _ = test.shape
    # refList = []
    # testList = []
    # for i in range(B):
    #     refVal = ref['original'][i][ref['gapStartIdx'][i] : ref['gapStartIdx'][i] + 1]
    #     testVal = test[i][ref['gapStartIdx'][i] : ref['gapStartIdx'][i] + 1]
    #     refList.append(refVal)
    #     testList.append(testVal)
    # finalRef = torch.stack(refList)
    # finalTest = torch.stack(testList)
    # boundaryLoss = lossObj(finalTest, finalRef.cuda().float())
    # loss = 0.67 * normalLoss + 0.33 * boundaryLoss
    # return loss

    # normalLoss = lossObj(test, ref['original'].cuda().float())
    B, _ = test.shape
    refList = []
    testList = []
    for i in range(B):
        refVal = ref['original'][i][ref['gapStartIdx'][i] : ref['gapEndIdx'][i]]
        testVal = test[i][ref['gapStartIdx'][i] : ref['gapEndIdx'][i]]
        refList.append(refVal)
        testList.append(testVal)
    finalRef = torch.stack(refList)
    finalTest = torch.stack(testList)
    loss = lossObj(finalTest, finalRef.cuda().float())
    return loss

if __name__ == "__main__":
    # setup logger
    setupLogger()
    # Load arguments
    allArgs = loadArguments()
    import pdb
    pdb.set_trace()
    print("HOOK")
    config = get_config(allArgs)
    print("MAIN WS: {}".format(config.MODEL.SWIN.WINDOW_SIZE))
    model = WaveformReconstructor(config, img_size=allArgs.img_size, num_classes=allArgs.num_classes).cuda()
    model.load_from(config)

    trainContainer = SeismicDataset(DATAPATH, gap_len=60, augment=True, dataType="train")
    testContainer = SeismicDataset(DATAPATH, gap_len=60, augment=False, dataType="test")
    trainGenerator = DataLoader(trainContainer,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS)
    testGenerator = DataLoader(testContainer,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=NUM_WORKERS)

    # Training
    model.train()
    lossObj = getLoss(lossName=allArgs.lossName)
    base_lr = allArgs.base_lr
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = allArgs.max_epochs
    max_iterations = allArgs.max_epochs * len(trainGenerator)
    # iterator = tqdm(range(max_epoch), ncols=70)
    lossVecE = []
    import pdb
    pdb.set_trace()
    for epochNum in range(max_epoch):
        eStart = time.time()
        lossVecB = []
        for batchIdx, batchData in enumerate(trainGenerator):
            if os.path.exists("pauseTraining.txt"):
                os.remove("pauseTraining.txt")
                import pdb
                pdb.set_trace()
            startTime = time.time()
            img, label = batchData
            img = img.cuda()
            img = img.float()
            op = model(img)
            # print("HOOK")
            import pdb
            pdb.set_trace()
            loss = computeLoss(lossObj, op, label)

            # loss = processLoss()
            lossVecB.append(loss.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            endTime = time.time()
            # print("Batch Time: {}".format(endTime - startTime))
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
        eEnd = time.time()
        lossEMean = sum(lossVecB) / float(len(lossVecB))
        print("Epoch Time {}: {} [Loss: {}]".format(epochNum + 1, eEnd - eStart, lossEMean))
        lossVecE.append(lossEMean)
        torch.save(model.state_dict(), "trainedModel.pt")
    import pdb
    pdb.set_trace()
    print("HOOK")

    torch.save(model.state_dict(), "trainedModel.pt")
    print("HOOK 2")

"""
(Pdb) orig = label['sample'][0] 
(Pdb) orig.shape
torch.Size([7200])
(Pdb) gap = img[0][0][0] 
(Pdb) gap.shape
torch.Size([7200])
(Pdb) recons = op[0] 
(Pdb) recons.shape
torch.Size([7200])
(Pdb) max(recons) 
tensor(0.0092, device='cuda:0', grad_fn=<UnbindBackward0>)
(Pdb) min(recons) 
tensor(-0.0081, device='cuda:0', grad_fn=<UnbindBackward0>)
(Pdb) max(orig) 
tensor(0.8456)
(Pdb) min(orig) 
tensor(-1.)
(Pdb) 1/0.0081
123.4567901234568
(Pdb) recons + recons * 120.0
tensor([ 0.3995,  0.5557, -0.0433,  ..., -0.3337,  0.1794,  0.2092],
       device='cuda:0', grad_fn=<AddBackward0>)
(Pdb) recons = recons * 120.0 
(Pdb) import copy
(Pdb) final = copy.deepcopy(orig) 
(Pdb) label['gapStartIdx'] 
tensor([6122, 4326, 6687, 6338])
(Pdb) label['gapEndIdx']   
tensor([6152, 4356, 6717, 6368])
(Pdb) final[6122:6152] = recons[6122:6152] 
(Pdb) from matplotlib import pyplot as plt
(Pdb) fig, ax = plt.subplots(3) 
(Pdb) ax[0].plot(orig[5500:6500]) 
[<matplotlib.lines.Line2D object at 0x000001867705BE80>]
(Pdb) ax[1].plot(gap.cpu()[5500:6500]) 
[<matplotlib.lines.Line2D object at 0x000001867705BE50>]
(Pdb) final = final.detach() 
(Pdb) ax[2].plot(final.cpu()[5500:6500]) 
[<matplotlib.lines.Line2D object at 0x00000186770710D0>]
(Pdb) plt.savefig("result.png") 

"""




