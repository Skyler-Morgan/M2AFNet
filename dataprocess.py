import argparse
import os
import natsort
import xlrd
import numpy as np
import imageio

from sklearn.model_selection import KFold

dict=['NORMAL', 'CNV', 'DR','AMD','CSC','RVO','OTHERS'] #前四个是3M中使用的，

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # other setting
    parser.add_argument('--data_path', type=str, default='/media/lq/One Touch/eyesdata/OCT500-ori/OCTA-300/', help='Raw data set folder')
    parser.add_argument('--modality_filename', type=list, default=['OCT', 'FULL'], help='Modalities used')
    parser.add_argument('--saveroot', type=str, default='../', help='Packed files after processing')

    opt = parser.parse_args()
    return opt

def read_excel(data_path):
    # 打开文件
    workBook = xlrd.open_workbook(r''+data_path+'Text labels.xlsx')

    sheet1_content1 = workBook.sheet_by_index(0)  # sheet索引从0开始
    # 3. sheet的名称，行数，列数
    print(sheet1_content1.name, sheet1_content1.nrows, sheet1_content1.ncols)

    # 4. 获取整行和整列的值（数组）
    id_cols = sheet1_content1.col_values(0)  # 获取第1列内容
    id_cols.pop(0) #去除标题
    id_cols= [int(i) for i in id_cols]
    id_cols = [str(i) for i in id_cols]
    disease_cols = sheet1_content1.col_values(4)  # 获取第5列内容
    disease_cols.pop(0)
    return id_cols, disease_cols
    #读图像并保存hdf5文件 self.data[modality_num,r,c,scan_num,ct_num] self.label[0,r,c,ct_num]

def read_imageme(datasetlist):
    data2d = np.zeros((1, 304, 304, len(datasetlist['FULL'])), dtype=np.int)
    print("picking ...It will take some minutes")

    ctlist = list(datasetlist['FULL'])  # 读取二维图像
    ct_num = -1
    for ct in ctlist:
        ct_num += 1
        labeladress = datasetlist['FULL'][ct]
        data2d[0, :, :, ct_num] = image_transform(labeladress)

    return data2d

# 读取图像并转换大小
def image_transform(filename): #对OCT sacan 压缩纵轴
    image = imageio.imread(filename)
    resize_image = np.array(image)
    return resize_image

def main():
    opt = parse_option()
    data_dir=opt.data_path
    modality=opt.modality_filename
    saveroot=opt.saveroot

    # 读取文件路径和标签信息
    id_cols, disease_cols = read_excel(data_dir)
    dedi = np.zeros(len(disease_cols))
    for id in range(len(disease_cols)):
        dedi[id] = dict.index(disease_cols[id])
    classnum = int(dedi.max() + 1)
    clist = []
    for i in range(classnum):
        clist.append([])

    for cnt, element in enumerate(dedi):
        clist[int(element)].append(cnt)

    datasetlist = {}
    for modal in modality:  # OCT/OCTA/Label
        datasetlist.update({modal: {}})  # update增加子列表

    for cnt, ct in enumerate(id_cols[:]):  # cnt表示索引，ct表示元素
        for modal in modality:
            if modal == 'OCT' or modality == 'OCTA':  # 读取三维数据
                datasetlist[modal].update({cnt: {}})
                scanlist = os.listdir(os.path.join(data_dir, modal, ct))  # 读取子文件夹下的所有ct切片数
                scanlist = natsort.natsorted(scanlist)
                for i in range(0, len(scanlist)):  # 读取子文件夹下的所有ct切片路径
                    scanlist[i] = os.path.join(data_dir, modal, ct, scanlist[i])
                datasetlist[modal][cnt] = scanlist  # 最后一个维度是保存list
            else:
                datasetlist[modal].update({cnt: {}})
                labeladdress = os.path.join(data_dir, modal, ct)
                datasetlist[modal][cnt] = labeladdress + '.bmp'  # 最后一个维度是保存单个文件路径

    # 根据多模态路径读取图像
    data2d = read_imageme(datasetlist)

    Bclist = []
    for i in range(classnum):
        Bzl = []
        for ct in clist[i]:  # cnt表示索引，ct表示元素
            for bct in datasetlist['OCT'][ct]:
                Bzl.append(bct)
        Bclist.append(Bzl)

    # 保存图像和标签
    np.savez(os.path.join(saveroot, "data3m.npz"), data2d, datasetlist, dedi, clist, Bclist)

    #生成训练数据集
    id_cols = np.array(id_cols)
    disease_cols = np.array(disease_cols)
    #5倍交叉验证
    kf = KFold(n_splits=5, shuffle=True)  # 初始化KFold
    for train_index, test_index in kf.split(id_cols):  # 调用split方法切分数据
        # print('train_index:%s , test_index: %s ' % (train_index, test_index))
        id_cols_train = id_cols[train_index]
        disease_cols_train = disease_cols[train_index]

    dedi = np.zeros(len(disease_cols_train))
    for id in range(len(disease_cols_train)):
        dedi[id] = dict.index(disease_cols_train[id])

    clist = []
    for i in range(classnum):
        clist.append([])

    for cnt, element in enumerate(dedi):
        clist[int(element)].append(cnt)

    datasetlist = {}
    for modal in modality:  # OCT/OCTA/Label
        datasetlist.update({modal: {}})  # update增加子列表

    for cnt, ct in enumerate(id_cols_train[:]):  # cnt表示索引，ct表示元素
        for modal in modality:
            if modal == 'OCT' or modality == 'OCTA':  # 读取三维数据
                datasetlist[modal].update({cnt: {}})
                scanlist = os.listdir(os.path.join(data_dir, modal, ct))  # 读取子文件夹下的所有ct切片数
                scanlist = natsort.natsorted(scanlist)
                for i in range(0, len(scanlist)):  # 读取子文件夹下的所有ct切片路径
                    scanlist[i] = os.path.join(data_dir, modal, ct, scanlist[i])
                datasetlist[modal][cnt] = scanlist  # 最后一个维度是保存list
            else:
                datasetlist[modal].update({cnt: {}})
                labeladdress = os.path.join(data_dir, modal, ct)
                datasetlist[modal][cnt] = labeladdress + '.bmp'  # 最后一个维度是保存单个文件路径

    # 根据多模态路径读取图像
    data2d = read_imageme(datasetlist)

    Bclist = []
    for i in range(classnum):
        Bzl = []
        for ct in clist[i]:  # cnt表示索引，ct表示元素
            for bct in datasetlist['OCT'][ct]:
                Bzl.append(bct)
        Bclist.append(Bzl)

    # 保存图像和标签
    np.savez(os.path.join(saveroot, "data3m_train.npz"), data2d, datasetlist, dedi, clist, Bclist)

if __name__ == '__main__':
    main()
