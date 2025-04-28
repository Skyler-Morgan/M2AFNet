import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as trans
import warnings
warnings.filterwarnings('ignore')
from lib.CFPOCT import build_mlp, fc
import lib.Head_loader2
import lib.readData
from sklearn.metrics import confusion_matrix
import itertools
from lib.CFPOCT import mam, mam3D
import lib.Bscan_loader2
import lib.Multi_loader2
import argparse
import torch.nn as nn

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

batchsize = 28
n_epochs = 100
init_lr = 1e-4
use_cuda = True
data_path = '/media/One Touch/eyesdata/OCT500-ori/OCTA-300/'#'E:\\task8\\OCT500\\OCTA-600\\'
modality_filename = ['OCT', 'FULL']
saveroot = '/media/One Touch/eyesdata/OCT500-ori/OCTA-300'#'D:/task8/logs'

linear_keyword = 'head'
tb_folder = '/home/project_dir/ViT-OCT-3M/save/Head'
#tensorboard
if not os.path.isdir(tb_folder):
    os.makedirs(tb_folder)

class_name = ['NORMAL', 'CNV', 'DR', 'AMD']
data2d, datasetlist, labs, clist, Bclist = lib.readData.read_datasetme('../data3m_train.npz')
f = np.load(os.path.join('/home/lq/project_dir/ViT-OCT-3M/save/Multi', "allmodallogits.npz"),allow_pickle=True)
octlog = f['arr_0']
cfplog = f['arr_1']
labs = f['arr_2']
oct_clip = f['arr_3']
cfp_clip = f['arr_4']

img_transforms = trans.Compose([
    trans.ToTensor(),
])

train_loader = lib.Head_loader2.DataLoader(
    dataset=[np.concatenate([octlog, cfp_clip], axis=1), np.concatenate([cfplog, oct_clip], axis=1), clist, labs],
    # dataset=[octlog, cfplog, clist, labs],
    img1_trans=img_transforms,
    img2_trans=img_transforms,
    batch_size=batchsize,
    num_workers=0
)

mlp1 = build_mlp(3, 1024, 1024, 512)
mlp2 = build_mlp(3, 1024, 1024, 512)
clh = fc(512,4)

mclh = Mamba(512)

files = os.listdir('save/Head/')  # 得到文件夹下的所有文件名称
for mdname in files: #遍历文件夹
    if mdname.endswith('.pth'):  # 如果检测到模型
        mdid = mdname.split('_')[2][:-4]
        path_experiment='save/Head/'+mdid+'/'

        if not os.path.isdir(path_experiment):
            os.makedirs(path_experiment)

        state_dict = torch.load('save/Head/'+mdname, map_location=torch.device('cpu'))
        msg = mlp1.load_state_dict(state_dict['mlp1'])
        msg = mlp2.load_state_dict(state_dict['mlp2'])
        msg = clh.load_state_dict(state_dict['clh'])
        msg = mclh.load_state_dict(state_dict['mclh'])
        device = torch.device("cuda:0" if use_cuda else "cpu")
        mlp1.to(device)
        mlp2.to(device)
        clh.to(device)
        mclh.to(device)
        mlp1.eval()
        mlp2.eval()
        clh.eval()
        mclh.eval()
        sm = torch.nn.Softmax(dim=1)    #沿通道方向
        gt = []
        predict = []
        with torch.no_grad():
            for i, data in enumerate(train_loader, 0):
                #读数据
                oct_imgs = data[0].to(device=device, dtype=torch.float32)
                cfp_imgs = data[1].to(device=device, dtype=torch.float32)
                labels = data[2].to(device=device, dtype=torch.long)
                bsz = labels.shape[0]
                oct_log = mlp1(oct_imgs)
                cfp_log = mlp2(cfp_imgs)
                log = torch.stack([oct_log, cfp_log], dim=1)
                log = mclh(log)
                log = sm(log)
                oct_ccon = log[:, 0, :]
                cfp_ccon = log[:, 1, :]
                oct_con = oct_ccon.mean(axis=1)
                cfp_con = cfp_ccon.mean(axis=1)
                flog = torch.mul(oct_ccon, oct_log)+torch.mul(cfp_ccon, cfp_log)
                pre = clh(flog)
                predicted = torch.max(pre.data, dim=1).indices
                for idx, _ in enumerate(labels):
                    gt.append(labels[idx])
                    predict.append(predicted[idx])

            ground_truth = np.array([g.item() for g in gt])
            prediction = np.array([pred.item() for pred in predict])

            title='Confusion matrix'
            cm = confusion_matrix(ground_truth, prediction)
            cmap=plt.cm.Blues
            print(cm)
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(class_name))
            plt.xticks(tick_marks, class_name, rotation=45)
            plt.yticks(tick_marks, class_name)
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(path_experiment + "confusion matrix.jpg", bbox_inches='tight')


            file_perf = open(path_experiment + 'performances.txt', 'w')
            print('class acc, sen, spe, pre, miou, f1')
            file_perf.write('class acc, sen, spe, pre, miou, f1' + '\n')
            n_classes = len(class_name)
            allacc = []
            allsen = []
            allspe = []
            allpre = []
            allmiou = []
            allf1 = []
            for i in range(n_classes):
                y_test = [int(x == i) for x in ground_truth]  # obtain binary label per class
                tn, fp, fn, tp = confusion_matrix(y_test, [int(x == i) for x in prediction]).ravel()
                acc = float(tp + tn) / (tn + fp + fn + tp)
                sen = float(tp) / (fn + tp)
                spe = float(tn) / (tn + fp)
                pre = float(tp) / (tp + fp)
                miou = float(tp) / (tp + fp + fn)
                f1 = 2 * pre * sen / (pre + sen)
                print(class_name[i], '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1))
                file_perf.write(class_name[i]+ '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (acc, sen, spe, pre, miou, f1) + '\n')
                allacc.append(acc)
                allsen.append(sen)
                allspe.append(spe)
                allpre.append(pre)
                allmiou.append(miou)
                allf1.append(f1)
            aacc = sum(allacc) / n_classes
            asen = sum(allsen) / n_classes
            aspe = sum(allspe) / n_classes
            apre = sum(allpre) / n_classes
            amiou = sum(allmiou) / n_classes
            af1 = sum(allf1) / n_classes
            print('mean_of_all', '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1))
            file_perf.write('mean_of_all' + '\t%.4f %.4f %.4f %.4f %.4f %.4f' % (aacc, asen, aspe, apre, amiou, af1) + '\n')

            file_perf.close()
            print("done")
