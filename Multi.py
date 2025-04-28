import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from lib.util import AverageMeter
from lib.util import adjust_learning_rate_sup
from lib.util import set_optimizer, save_model
from lib.mlosses import SupConLoss
import lib.readData
import torchvision.transforms as trans
from lib.Multi_loader import DataLoader
from lib.Multi_loader2 import DataLoader2
from lib.CFPOCT import mam, mam3D
import clip


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--saveroot', type=str, default='../data3m_train.npz',
                        help='Packed files after processing')
    parser.add_argument('--batchsize', type=int, default=40,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=50,
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

    # other setting
    opt = parser.parse_args()

    opt.tb_folder = './save/Multi'
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder =  './save/Multi'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model1 = mam3D(
        num_patches=304,
        patch_dim=512,
        dim=768,
        outdim = 512,
        emb_dropout=0.1
    )
    model2 = mam(
    image_size=304,
    patch_size=38,
    dim=768,
    channels=1,
    outdim= 512,
    emb_dropout = 0.1)

    model3, preprocess = clip.load("ViT-B/32", device="cuda")
    model3.eval()

    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():

        model1 = model1.cuda()
        model2 = model2.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model1, model2,  criterion

def train(train_loader, model1, model2, criterion, optimizer, epoch):
    """one epoch training"""
    model1.train()
    model2.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if (idx+1)%2 == 1:
            images1 = data[0].cuda(non_blocking=True)
            labels1 = data[2].cuda(non_blocking=True)
        else:
            data_time.update(time.time() - end)

            images2 = data[1].cuda(non_blocking=True)
            labels2 = data[2].cuda(non_blocking=True)
            bsz = labels2.shape[0]

            f1 = model1(images1)
            with torch.no_grad():
                f2 = model2(images2)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            labels = torch.cat([labels1, labels2])
            loss = criterion(features, labels)  # [28，2，256] [56]
            losses.update(loss.item(), bsz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def train2(train_loader, model1, model2, criterion, optimizer, epoch):
    """one epoch training"""
    model1.eval()
    model2.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if (idx + 1) % 2 == 1:
            images1 = data[0].cuda(non_blocking=True)
            labels1 = data[2].cuda(non_blocking=True)
        else:
            data_time.update(time.time() - end)
            images2 = data[1].cuda(non_blocking=True)
            labels2 = data[2].cuda(non_blocking=True)
            bsz = labels2.shape[0]

            with torch.no_grad():
                f1 = model1(images1)
            f2 = model2(images2)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            labels = torch.cat([labels1, labels2])
            loss = criterion(features, labels)
            losses.update(loss.item(), bsz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg

def main():
    opt = parse_option()

    # 读取所有数据
    data2d, datasetlist, labs, clist, Bclist = lib.readData.read_datasetme(opt.saveroot)  # 读取文件路径和诊断真值
    f = np.load(os.path.join('save/Bscan', "datalogits.npz"), allow_pickle=True)
    datalogits = f['arr_0'][()]
    augmentation = trans.Compose([
        trans.ToTensor(),
    ])

    # 设置训练数据加载器
    train_loader = DataLoader(
        dataset=[datasetlist, labs, clist, datalogits],
        img1_trans=augmentation,
        img2_trans=augmentation,
        batch_size=opt.batchsize,
        num_workers=0
    )

    model1, model2,criterion = set_model(opt)

    optimizer1 = set_optimizer(opt, model1)
    optimizer2 = set_optimizer(opt, model2)

    pretrained = r'save\OCT\last.pth'
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['model']

        msg = model1.load_state_dict(state_dict, strict=False)
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

    pretrained = r'save\CFP\best.pth'
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['model']
            msg = model2.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))


    logger = SummaryWriter(opt.tb_folder)  # tensorboard初始化一个写入单元

    bestloss1 = None
    bestloss2 = None
    start_time = time.time()  # 记录开始训练时间
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate_sup(opt, optimizer1)
        adjust_learning_rate_sup(opt, optimizer2)
        time1 = time.time()
        loss1 = train(train_loader, model1, model2, criterion, optimizer1, epoch)
        if bestloss1 == None:
            bestloss1 = loss1
        else:
            if loss1<=bestloss1:
                bestloss1 = loss1
                save_file = os.path.join(
                    opt.save_folder, 'best_ViT3D.pth')
                save_model(model1, save_file)

        loss2 = train2(train_loader, model1, model2, criterion, optimizer2, epoch)
        if bestloss2 == None:
            bestloss2 = loss2
        else:
            if loss2<=bestloss2:
                bestloss2=loss2
                save_file = os.path.join(
                    opt.save_folder, 'best_ViT.pth')
                save_model(model2, save_file)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        logger.add_scalar('loss1', loss1, epoch)
        logger.add_scalar('loss2', loss2, epoch)
        logger.add_scalar('learning_rate', optimizer1.param_groups[0]['lr'], epoch)

    end_time = time.time()  # 记录结束训练时间
    print('cost %f second' % (end_time - start_time))

    test_loader = DataLoader2(
        dataset=[datasetlist, labs, clist, datalogits],
        img1_trans= augmentation,
        img2_trans= augmentation,
        batch_size=opt.batchsize,
        num_workers=0,
        splfun='Sequen'
    )

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    octlog = []
    cfplog = []
    labs = []

    model1.eval()
    model2.eval()

    oct_clip_log = []
    cfp_clip_log = []
    model3, preprocess = clip.load("ViT-B/32", device=device)
    model3 = model3.to(device)
    model3.eval()

    device = torch.device("cuda:0")
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            oct_imgs = data[0].to(device=device, dtype=torch.float32)
            cfp_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].cpu().numpy()
            logits1 = model1(oct_imgs).cpu().numpy()
            logits2 = model2(cfp_imgs).cpu().numpy()

            oct_clip = model3.encode_image(data[3].to(device)).cpu().numpy()
            cfp_clip = model3.encode_image(data[4].to(device)).cpu().numpy()

            for ib in range(len(labels)):
                octlog.append(logits1[ib])
                cfplog.append(logits2[ib])
                labs.append(labels[ib])

                oct_clip_log.append(oct_clip[ib])
                cfp_clip_log.append(cfp_clip[ib])

    oct_clip_log = np.stack(oct_clip_log, 0)
    cfp_clip_log = np.stack(cfp_clip_log, 0)
    octlog = np.stack(octlog)
    cfplog = np.stack(cfplog)
    labs = np.stack(labs)

    oct_clip_log = np.stack(oct_clip_log)
    cfp_clip_log = np.stack(cfp_clip_log)

    np.savez(os.path.join(opt.save_folder, "allmodallogits.npz"), octlog, cfplog, labs, oct_clip_log, cfp_clip_log)


if __name__ == '__main__':
    main()
