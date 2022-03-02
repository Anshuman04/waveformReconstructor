import os
import sys
import logging
import argparse

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss

from customDataset import SeismicDataset
from defaultConfig import get_config
from model import WaveformReconstructor


BATCH_SIZE = 4
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

if __name__ == "__main__":
    # setup logger
    setupLogger()
    # Load arguments
    allArgs = loadArguments()
    # import pdb
    # pdb.set_trace()
    print("HOOK")
    config = get_config(allArgs)
    print("MAIN WS: {}".format(config.MODEL.SWIN.WINDOW_SIZE))
    model = WaveformReconstructor(config, img_size=allArgs.img_size, num_classes=allArgs.num_classes).cuda()
    model.load_from(config)

    trainContainer = SeismicDataset(DATAPATH, gap_len=30, augment=True, dataType="train")
    testContainer = SeismicDataset(DATAPATH, gap_len=30, augment=False, dataType="test")
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
    ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=allArgs.base_lr, momentum=0.9, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = allArgs.max_epochs
    max_iterations = allArgs.max_epochs * len(trainGenerator)
    # iterator = tqdm(range(max_epoch), ncols=70)
    for epochNum in range(max_epoch):
        for batchIdx, batchData in enumerate(trainGenerator):
            import pdb
            pdb.set_trace()
            img, label = batchData
            img = img.cuda()
            op = model(img)
            print("HOOK")



