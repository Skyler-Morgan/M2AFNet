import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

from lib.util import TwoCropTransform, AverageMeter
from lib.util import set_optimizer, save_model
#from networks.resnet_big import SupConResNet
from lib.losses import SupConLoss
import lib.readData
import torchvision.transforms as trans
from lib.OCT_loader import DataLoader
from lib.CFPOCT import mam3D

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--saveroot', type=str, default='../data3m_train.npz',
                        help='Packed files after processing')
    parser.add_argument('--batch_size', type=int, default=40,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    opt = parser.parse_args()
    opt.tb_folder = './save/OCT'
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder =  './save/OCT'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = mam3D(
        num_patches=304,
        patch_dim=512,
        dim=768,
        outdim = 512,
        emb_dropout=0.1
    )
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([data[0], data[0]], dim=0)

        images = images.cuda(non_blocking=True) #对图像进行增强生成两个批次
        labels = data[1].cuda(non_blocking=True)

        bsz = labels.shape[0]

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
               epoch, idx + 1, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses))
        sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # 读取所有数据
    data2d, datasetlist, labs, clist, Bclist = lib.readData.read_datasetme(opt.saveroot)  # 读取文件路径和诊断真值
    f = np.load(os.path.join('save/Bscan', "datalogits.npz"), allow_pickle=True)
    datalogits = f['arr_0'][()] # ndarray转化为内置字典类型dict
    # 训练和评估双模态图像变换
    augmentation = trans.Compose([
        trans.ToTensor(),
    ])

    # 设置训练数据加载器
    train_loader = DataLoader(
        dataset=[datasetlist, labs, clist, datalogits],
        img1_trans=augmentation,
        img2_trans=augmentation,
        batch_size=opt.batch_size,
        num_workers=0
    )
    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    logger = SummaryWriter(opt.tb_folder)  # tensorboard初始化一个写入单元
    start_time = time.time()  # 记录开始训练时间
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate_sup(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.add_scalar('loss', loss, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        save_file = os.path.join(
            opt.save_folder, 'best.pth')
        save_model(model, save_file)

    end_time = time.time()  # 记录结束训练时间
    print('cost %f second' % (end_time - start_time))

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, save_file)


if __name__ == '__main__':
    main()
