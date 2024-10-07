import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def normalized_thresh(z, mu=1.0):
    if len(z.shape) == 1:
        mask = (torch.norm(z, p=2, dim=0) < np.sqrt(mu)).float()
        return mask * z + (1 - mask) * F.normalize(z, dim=0) * np.sqrt(mu)
    else:
        mask = (torch.norm(z, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        return mask * z + (1 - mask) * F.normalize(z, dim=1) * np.sqrt(mu)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)



class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, args=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.mu = 1.0

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        
        out = out.view(-1, self.nChannels) # 64 x 128
        return self.fc(out)

    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        
        out = out.view(-1, self.nChannels)
        return out
    
    def feature_list(self, x):
        out_list = [] 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list
    
    def sup_D(self, z1, z2, uz1, uz2, target, mu=1.0):
        device = z1.device
        bsz_l, bsz_u = len(z1), len(uz1)

        mat_ll = torch.matmul(z1, z2.T)
        mat_uu = torch.matmul(uz1, uz2.T)

        mat_lu_s2 = torch.matmul(z1, uz2.T) ** 2
        mat_ul_s2 = torch.matmul(uz1, z2.T) ** 2
        mat_ll_s2 = mat_ll ** 2 * (1 - torch.diag(torch.ones(bsz_l)).to(device))
        mat_uu_s2 = mat_uu ** 2 * (1 - torch.diag(torch.ones(bsz_u)).to(device))

        c1, c2 = self.args.c1, self.args.c2
        c3, c4, c5 = self.args.c3, self.args.c4, self.args.c5

        target_ = target.contiguous().view(-1, 1)
        pos_labeled_mask = torch.eq(target_, target_.T).to(device)
        cls_sample_count = pos_labeled_mask.sum(1)

        loss1 = - c1 * torch.sum((mat_ll * pos_labeled_mask) / cls_sample_count ** 2)

        pos_unlabeled_mask = torch.diag(torch.ones(bsz_u)).to(device)
        loss2 = - c2 * torch.sum(mat_uu * pos_unlabeled_mask) / bsz_u

        loss3 = c3 * torch.sum(mat_ll_s2 / (cls_sample_count[:, None] * cls_sample_count[None, :]))

        loss4 = c4 * torch.sum(mat_lu_s2 / (cls_sample_count[:, None] * bsz_u)) + \
                c4 * torch.sum(mat_ul_s2 / (cls_sample_count[None, :] * bsz_u))

        loss5 = c5 * torch.sum(mat_uu_s2) / (bsz_u * (bsz_u - 1))

        return (loss1 + loss2 + loss3 + loss4 + loss5) / mu, {"loss1": loss1 / mu, "loss2": loss2 / mu,
                                                              "loss3": loss3 / mu, "loss4": loss4 / mu,
                                                              "loss5": loss5 / mu}
    
    def un_D(self, z1, z2, mu=1.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
        z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                    z1.shape[0] / (z1.shape[0] - 1)
        return (loss_part1 + loss_part2) / mu, {"loss2": loss_part1 / mu, "loss5": loss_part2 / mu}

    def forward_backbone(self, x, return_feat=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels) # 64 x 128
        if return_feat:
            return self.fc(out), out
        else:
            return out
        
    def forward_scl(self, x1, x2, ux1, ux2, target=None, mu=1.0):
        x1 = torch.cat([x1, ux1], 0)
        x2 = torch.cat([x2, ux2], 0)
        # f1, f2 = self.backbone(x1), self.backbone(x2)
        # z1, z2 = self.projector(f1), self.projector(f2)
        z1 = self.forward_backbone(x1)
        z2 = self.forward_backbone(x2)
        L, d_dict = self.un_D(z1, z2, mu=self.mu)
        return {'loss': L, 'd_dict': d_dict}

    def forward_sscl(self, x1, x2, ux1, ux2, target, mu=1.0):
        x = torch.cat([x1, x2, ux1, ux2], 0)
        proj_feat = normalized_thresh(self.forward_backbone(x))
        z = proj_feat

        z1 = z[0:len(x1), :]
        z2 = z[len(x1):len(x1)+len(x2), :]
        uz1 = z[len(x1)+len(x2):len(x1)+len(x2)+len(ux1), :]
        uz2 = z[len(x1)+len(x2)+len(ux1):, :]

        spec_loss, d_dict = self.sup_D(z1, z2, uz1, uz2, target, mu=self.mu)
        loss = spec_loss
        return {'loss': loss, 'd_dict': d_dict}
