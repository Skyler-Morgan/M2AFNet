import os
import numpy as np
import torch
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')
from lib.CFPOCT import mam, Twoenc
from lib.Bscan_loader import DataLoader
import lib.Bscan_loader2
import lib.readData
from functools import partial
import math
import time
import argparse

def parse_option():
    parser = argparse.ArgumentParser(description='Argument for Bscan training')
    parser.add_argument('--saveroot', type=str, default='../data3m_train.npz',
                        help='Packed files after processing')
    parser.add_argument('--image_size', type=int, default=304,
                        help='input image size')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batchsize', default=320, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                             'batch size of all GPUs on all nodes when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    # moco specific configs:
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')

    opt = parser.parse_args()

    opt.save_folder =  './save/Bscan'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt

def train(train_loader, model, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        train_loader.len,
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    moco_m = args.moco_m
    for i, images in enumerate(train_loader, 0):  # 从0开始索引
        # measure data loading time
        data_time.update(time.time() - end)

        lr = args.lr
        learning_rates.update(lr)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m

def main():
    opt = parse_option()

    # 读取所有数据
    data2d, datasetlist, labs, clist, Bclist = lib.readData.read_datasetme(opt.saveroot)  # 读取文件路径和诊断真值

    augmentation1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
            opt.image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    ])

    augmentation2 = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 设置训练数据加载器
    train_loader = DataLoader(
        dataset=[datasetlist, labs, clist, Bclist],
        img1_trans=augmentation1,
        img2_trans=augmentation2,
        batch_size=opt.batchsize,
        num_workers=0
    )

    #定义模型 优化器 准则
    model = Twoenc(partial(mam,
        image_size = 304,
        patch_size = 38,
        dim=768,
        channels=1,
        depth = 6,
        heads = 6,
        mlp_dim = 1024,
        outdim= 512,
        dropout = 0.1,
        emb_dropout = 0.1),    #调用函数
        512, 1024, 0.2)
    device = torch.device("cuda:0" )
    model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr,weight_decay=1e-6)

    scaler = torch.cuda.amp.GradScaler()#梯度缩放防止小梯度消失

    #训练
    start_time = time.time()  # 记录开始训练时间
    for epoch in range(opt.epochs):
        loss = train(train_loader, model, optimizer, scaler, epoch, opt)
        print('epoch: %04d, loss: %f' % (epoch, loss))
        if epoch == 0:  # only the first GPU saves checkpoint
            minloss = loss
        elif loss <= minloss:
            minloss = loss
            torch.save({
                'state_dict': model.state_dict(),
            }, os.path.join(opt.save_folder,'model_best.pth.tar'))
            print("save best")
    end_time = time.time()  # 记录结束训练时间
    print('cost %f second' % (end_time - start_time))

    #dictionary
    pretrained = os.path.join(opt.save_folder,'model_best.pth.tar')
    model = mam(
        image_size = 304,
        patch_size = 38,
        dim=768,
        channels=1,
        outdim= 512,)

    # load from pre-trained, before DistributedDataParallel constructor
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # print(k)
                # retain only base_encoder up to before the embedding layer 只保留base_encoder，并且去掉head
                if k.startswith('module.momentum_encoder'):
                    # remove prefix
                    state_dict[k[len("module.momentum_encoder."):]] = state_dict[k]  # 只保留base_encoder.之后的有效字
                # delete renamed or unused k
                del state_dict[k]  # 删除列表中的元素

            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    # model = baidu_lib.Model()
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model)
    model.to(device)

    augmentation = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_loader = lib.Bscan_loader2.DataLoader(
        dataset=[datasetlist, labs, clist, Bclist],
        img1_trans=augmentation,
        img2_trans=augmentation,
        batch_size=opt.batchsize,
        num_workers=0,
        splfun='suqs',
    )

    datalogits = {}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # load batch
            names = data[0]
            oct_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].cpu().numpy()
            logits = model(oct_imgs).cpu().numpy()
            for ib in range(opt.batchsize):
                datalogits.update({names[ib]: logits[ib, :]})  # update增加子列表
    np.savez(os.path.join(opt.save_folder, "datalogits.npz"), datalogits)

if __name__ == '__main__':
    main()
