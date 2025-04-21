import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
###########################################
# 1. Split-Attention模块相关定义
###########################################

class rSoftMax(nn.Module):
    """
    rSoftMax: 对 Split-Attention 的输出进行 softmax 归一化
    """
    def __init__(self, radix, groups):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.groups = groups
        
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            # x shape: [B, groups*radix*C, 1, 1] -> reshape to [B, groups, radix, C]
            x = x.view(batch, self.groups, self.radix, -1)
            x = torch.softmax(x, dim=2)
            # reshape回 [B, groups*radix*C, 1, 1]
            x = x.view(batch, -1, 1, 1)
        else:
            x = torch.sigmoid(x)
        return x

class SplAtConv2d(nn.Module):
    """
    SplAtConv2d: 实现 Split-Attention 卷积。该模块先做普通卷积（注意 groups 设为 groups*radix），
    再对结果进行全局聚合，最后计算 attention 权重并融合各个分支。
    """
    def __init__(self, in_channels, channels, kernel_size, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, reduction_factor=4,
                 norm_layer=nn.BatchNorm2d):
        super(SplAtConv2d, self).__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.radix = radix
        self.channels = channels
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride,
                              padding, dilation, groups=groups * radix, bias=bias)
        self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels, inter_channels, kernel_size=1, groups=groups)
        self.bn1 = norm_layer(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, kernel_size=1, groups=groups)
        self.rsoftmax = rSoftMax(radix, groups)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)
        
        batch, channels_radix, H, W = x.size()
        if self.radix > 1:
            # reshape后求和聚合各个分支
            x = x.view(batch, self.radix, self.channels, H, W)
            gap = x.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu1(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        if self.radix > 1:
            atten = atten.view(batch, self.radix, self.channels, 1, 1)
            # 对各个分支进行加权求和
            x = x * atten
            x = x.sum(dim=1)
        else:
            x = x * atten
        return x

###########################################
# 2. ResNeSt Bottleneck模块定义
###########################################

class ResNeStBottleneck(nn.Module):
    """
    ResNeStBottleneck: 使用 SplAtConv2d 来实现中间3x3卷积的 Split-Attention 机制。
    与传统 Bottleneck 相比，该模块在第二层卷积处做了改动。
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 radix=2, groups=1, bottleneck_width=64, norm_layer=nn.BatchNorm2d):
        super(ResNeStBottleneck, self).__init__()
        width = int(planes * (bottleneck_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = SplAtConv2d(width, width, kernel_size=3, stride=stride,
                                  padding=dilation, dilation=dilation, groups=groups,
                                  bias=False, radix=radix, reduction_factor=4, norm_layer=norm_layer)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

###########################################
# 3. 使用 ResNeStBottleneck 构造骨干网络 (ResNeSt50)
###########################################

class ModifiedResNeSt50(nn.Module):
    def __init__(self, radix=2, groups=1, bottleneck_width=64, 
                 replace_stride_with_dilation=[False, True, True],
                 norm_layer=nn.BatchNorm2d):
        super(ModifiedResNeSt50, self).__init__()
        self.inplanes = 64
        # Stem 部分（与 ResNet 类似）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 分层构造：layer1 ~ layer4
        self.layer1 = self._make_layer(ResNeStBottleneck, 64, 3, stride=1, dilation=1,
                                       radix=radix, groups=groups, bottleneck_width=bottleneck_width,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(ResNeStBottleneck, 128, 4, stride=2,
                                       dilation=1 if not replace_stride_with_dilation[0] else 2,
                                       radix=radix, groups=groups, bottleneck_width=bottleneck_width,
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(ResNeStBottleneck, 256, 6, stride=2,
                                       dilation=1 if not replace_stride_with_dilation[1] else 2,
                                       radix=radix, groups=groups, bottleneck_width=bottleneck_width,
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(ResNeStBottleneck, 512, 3, stride=2,
                                       dilation=1 if not replace_stride_with_dilation[2] else 2,
                                       radix=radix, groups=groups, bottleneck_width=bottleneck_width,
                                       norm_layer=norm_layer)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    radix=2, groups=1, bottleneck_width=64, norm_layer=nn.BatchNorm2d):
        downsample = None
        previous_dilation = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=previous_dilation,
                            downsample=downsample, radix=radix, groups=groups,
                            bottleneck_width=bottleneck_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation,
                                radix=radix, groups=groups, bottleneck_width=bottleneck_width,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        return low_level_feat, x

###########################################
# 4. ASPP、Decoder、BEM 等模块（保持不变）
###########################################

# ASPP相关模块
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ])
        for rate in atrous_rates:
            self.branches.append(ASPPConv(in_channels, out_channels, rate))
        self.branches.append(ASPPPooling(in_channels, out_channels))
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    def forward(self, x):
        res = [branch(x) for branch in self.branches]
        return self.project(torch.cat(res, dim=1))

# Decoder模块
class Decoder(nn.Module):
    def __init__(self, low_level_in, low_level_out, out_channels, num_classes):
        super(Decoder, self).__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_level_in, low_level_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_out),
            nn.ReLU(inplace=True)
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(low_level_out + out_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    def forward(self, low_feat, x):
        low_feat = self.low_conv(low_feat)
        x = F.interpolate(x, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_feat], dim=1)
        return self.last_conv(x)

# 边界增强模块BEM（保持不变）
class BEM(nn.Module):
    def __init__(self, in_channels):
        super(BEM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

###########################################
# 5. 整体 DeepLabV3+ 模型定义
###########################################

class DeepLabV3Plus_Advanced(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus_Advanced, self).__init__()
        # 使用包含 Split-Attention 的 ResNeSt50 作为骨干网络
        self.backbone = ModifiedResNeSt50()
        self.bem = BEM(in_channels=2048)
        self.aspp = ASPP(in_channels=2048, out_channels=384, atrous_rates=[12, 24, 36])
        self.decoder = Decoder(low_level_in=256, low_level_out=48, out_channels=384, num_classes=num_classes)
    def forward(self, x):
        input_size = x.shape[-2:]
        low_feat, high_feat = self.backbone(x)
        high_feat = self.bem(high_feat)
        aspp_out = self.aspp(high_feat)
        seg_out = self.decoder(low_feat, aspp_out)
        seg_out = F.interpolate(seg_out, size=input_size, mode='bilinear', align_corners=False)
        return seg_out, seg_out
