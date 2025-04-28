"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # len(features.shape)=3 [batch,2,dim] 不经过以下处理
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3: #只保留前两维，之后统一算特征维度
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) #复制tensor并断开这两个变量之间的依赖 [16]->[16,1]
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)    #构建标签相似矩阵
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  #记录增强几次
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #将两种增强图像特征连接成2N
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature   #复制一份特征
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits    点积计算相似度 （zi.zp/t）
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature*1000)
        # for numerical stability 对角线为最大值相减成0，其余全为负值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)    #为增强图像扩充标签矩阵
        # mask-out self-contrast cases 生成对角线为0的矩阵
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask   #计算正对，清除对角线的影响，自己对自己

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask    #exp(zi.zp/t) 分母 所有exp逻辑值
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  #前一项log(exp(zi.zp/t))=(zi.zp/t) 是分子 后一项是分母求和

        # compute mean of log-likelihood over positive mask * log_prob->只计算正对 p属于P(i) / mask.sum(1)是/P(i)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos #取反
        loss = loss.view(anchor_count, batch_size).mean()   #求平均，对于batch和增强的倍数2

        return loss
