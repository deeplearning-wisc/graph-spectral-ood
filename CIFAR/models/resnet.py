import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from wrn import normalized_thresh
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        #now out dim: batch_size, 512, 1, 1
        out = torch.flatten(out, 1) #start_dim = 1
        return out

    # function to extact a specific feature
    def intermediate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)              
        return out

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_list.append(out)
        return out_list


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet18', normalize = False,  num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        self.normalize = normalize

    def forward(self, x):
        features = self.encoder(x)
        if self.normalize: 
            features =  F.normalize(features, dim=1)
        return self.fc(features)

class SupCEHeadResNet(nn.Module):
    """encoder + head"""
    def __init__(self, args, multiplier = 1):
        super(SupCEHeadResNet, self).__init__()
        self.args = args
        model_fun, dim_in = model_dict[args.model]
        #if args.in_dataset == 'ImageNet-100':
        #if args.dataset == 'ImageNet-100':
        if args.model == 'resnet50':
            model = models.resnet50(pretrained=True)
            #for name, p in model.named_parameters():
            #    if not name.startswith('layer4'):
            #        p.requires_grad = False
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    #if hasattr(module, 'weight'):
                    #    module.weight.requires_grad_(False)
                    #if hasattr(module, 'bias'):
                    #    module.bias.requires_grad_(False)
                    module.eval()
            modules=list(model.children())[:-1] # remove last linear layer
            self.encoder =nn.Sequential(*modules)
        elif args.model == 'resnet34':
            model = models.resnet34(pretrained=True)
            for name, p in model.named_parameters():
                if not name.startswith('layer4'):
                    p.requires_grad = False
            modules=list(model.children())[:-1] # remove last linear layer
            self.encoder =nn.Sequential(*modules)
        else:
            self.encoder = model_fun() #cifar10
            #model = models.resnet50(pretrained=True)
            #dim_in = model.fc.in_features
            #modules=list(model.children())[:-1] # remove last linear layer
            #self.encoder =nn.Sequential(*modules)

        #if args.dataset == 'PACS':
        #self.fc = nn.Linear(dim_in, 7) # 7
        #else:
        self.fc = nn.Linear(dim_in, 100) # 7
        self.multiplier = multiplier
        

        self.dim_in = dim_in
        #self.dropout = nn.Dropout(0.5)        
        '''
        if args.head == 'linear':
            self.head = nn.Sequential(nn.Linear(dim_in, args.feat_dim)) #cartoon checkpoints
            #self.head = nn.Linear(dim_in, args.feat_dim)
        elif args.head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, args.feat_dim)
            )
        '''
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(-1, self.dim_in)
        # unnorm_features = self.head(feat) #.view(-1, 128)
        # features= F.normalize(unnorm_features, dim=1)
        output = self.fc(feat)
        return output
    
    def intermediate_forward(self, x):
        feat = self.encoder(x).squeeze()
        return F.normalize(feat, dim=1)

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
        '''
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        '''
        x = self.encoder(x)
        out = x.view(-1, self.dim_in)
    
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
        L, d_dict = self.un_D(z1, z2, mu=mu)
        return {'loss': L, 'd_dict': d_dict}

    def forward_sscl(self, x1, x2, ux1, ux2, target, mu=1.0):
        x = torch.cat([x1, x2, ux1, ux2], 0)
        proj_feat = normalized_thresh(self.forward_backbone(x))
        z = proj_feat

        z1 = z[0:len(x1), :]
        z2 = z[len(x1):len(x1)+len(x2), :]
        uz1 = z[len(x1)+len(x2):len(x1)+len(x2)+len(ux1), :]
        uz2 = z[len(x1)+len(x2)+len(ux1):, :]

        spec_loss, d_dict = self.sup_D(z1, z2, uz1, uz2, target, mu=mu)
        loss = spec_loss
        return {'loss': loss, 'd_dict': d_dict}
