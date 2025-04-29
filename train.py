import os
import numpy as np
from lib.util import AverageMeter, adjust_learning_rateme
import torch
import torch.nn as nn
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')
from lib.CFPOCT import build_mlp, fc
from lib.Head_loader import DataLoader
import lib.readData
from tensorboardX import SummaryWriter
import time
import argparse
from mamba_ssm.modules.mamba_simple import Mamba
class Mamba(nn.Module):
    def __init__(self, dim):
        super(Mamba, self).__init__()
        self.encoder = Mamba(dim)
        self.conv33conv33conv11 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(dim)
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--saveroot', type=str, default='../data3m_train.npz',
                        help='Packed files after processing')
    parser.add_argument('--batchsize', type=int, default=40,
                        help='batch_size')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='learning rate')

    # other setting
    opt = parser.parse_args()

    opt.tb_folder = './save/Head'
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder =  './save/Head'
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def main():
    opt = parse_option()
    logger = SummaryWriter(opt.tb_folder)
    _, _, _, clist, _ = lib.readData.read_datasetme(opt.saveroot)  # 读取文件路径和诊断真值
    f = np.load(os.path.join('save/Multi', "allmodallogits.npz"),allow_pickle=True)
    octlog = f['arr_0']
    cfplog = f['arr_1']
    labs = f['arr_2']
    oct_clip = f['arr_3']
    cfp_clip = f['arr_4']

    img_transforms = trans.Compose([
        trans.ToTensor(),
    ])

    train_loader = DataLoader(
        dataset=[np.concatenate([octlog, cfp_clip], axis=1), np.concatenate([cfplog, oct_clip], axis=1), clist, labs],
        img1_trans=img_transforms,
        img2_trans=img_transforms,
        batch_size=opt.batchsize,
        num_workers=0
    )

    mlp1 = build_mlp(3, 1024, 1024, 512)
    mlp2 = build_mlp(3, 1024, 1024, 512)
    clh = fc(512,4)
    mclh = Mamba(512)
    device = torch.device("cuda:0")
    mlp1.to(device)
    mlp2.to(device)
    clh.to(device)
    mclh.to(device)

    opt1 = torch.optim.Adam(mlp1.parameters(), lr=opt.init_lr)
    opt2 = torch.optim.Adam(mlp2.parameters(), lr=opt.init_lr)
    opt3 = torch.optim.Adam(clh.parameters(), lr=opt.init_lr)
    opt4 = torch.optim.Adam(mclh.parameters(), lr=opt.init_lr)

    criterion = nn.CrossEntropyLoss()

    adlr = adjust_learning_rateme(opt.init_lr*2, opt.n_epochs/2, opt.n_epochs)

    mlp1.train()
    mlp2.train()
    clh.train()
    mclh.train()
    sm = torch.nn.Softmax(dim=1)
    start_time = time.time()  # 记录开始训练时间
    bestloss = None
    for epoch in range(opt.n_epochs):
        if epoch == 0:
            ocon=0
            ccon=0
        else:
            ocon = cfpcon.avg / (octcon.avg + cfpcon.avg)
            ccon = octcon.avg / (octcon.avg + cfpcon.avg)

        losses = AverageMeter()
        octcon = AverageMeter()
        cfpcon = AverageMeter()

        olr = adlr.update(epoch, ocon, opt1)
        clr = adlr.update(epoch, ccon, opt2)
        elr = adlr.update(epoch, 0.5, opt3)
        mlr = adlr.update(epoch, 0.5, opt4)
        for i, data in enumerate(train_loader, 0):
            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            opt4.zero_grad()

            oct_imgs = data[0].to(device=device, dtype=torch.float32)
            cfp_imgs = data[1].to(device=device, dtype=torch.float32)
            labels = data[2].to(device=device, dtype=torch.long)
            bsz = labels.shape[0]
            oct_log = mlp1(oct_imgs)
            cfp_log = mlp2(cfp_imgs)
            log = torch.stack([oct_log,cfp_log],dim=1)
            log = mclh(log)
            log = sm(log)
            oct_ccon = log[:, 0, :]
            cfp_ccon = log[:, 1, :]
            oct_con = oct_ccon.mean(axis=1)
            cfp_con = cfp_ccon.mean(axis=1)
            octcon.update(oct_con.cpu().detach().numpy().mean(),bsz)
            cfpcon.update(cfp_con.cpu().detach().numpy().mean(),bsz)
            flog = torch.mul(oct_ccon, oct_log)+torch.mul(cfp_ccon, cfp_log)
            pre = clh(flog)
            loss = criterion(pre, labels)
            supl = 0.006*epoch+0.3  
            loss = (1-supl)*la+supl*lc  
            losses.update(loss.item(), bsz)
            loss.backward()
            opt1.step()
            opt2.step()
            opt3.step()
            opt4.step()

        logger.add_scalar('loss', losses.avg, epoch)
        logger.add_scalar('octcon', octcon.avg, epoch)
        logger.add_scalar('cfpcon', cfpcon.avg, epoch)
        logger.add_scalar('lr', elr, epoch)
        logger.add_scalar('olr', olr, epoch)
        logger.add_scalar('clr', clr, epoch)

        print(losses.avg)
        if bestloss == None:
            bestloss = losses.avg
        else:
            if losses.avg <= bestloss:
                bestloss = losses.avg
                save_file = os.path.join(
                    'save/Head','best_model_{:.4f}'.format(bestloss)+ '.pth')
                print('==> Saving...')
                mlp1.train()
                mlp2.train()
                clh.train()
                state = {
                    'mlp1': mlp1.state_dict(),
                    'mlp2': mlp2.state_dict(),
                    'clh': clh.state_dict(),
                }
                torch.save(state, save_file)
                del state
    end_time = time.time()
    print('cost %f second' % (end_time - start_time))

if __name__ == '__main__':
    main()
