import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
import torchvision.models as models

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x
class _PF(nn.Module):
  def __init__(self, dim_pv, kernel_size=3, padding=1, reduction=4):
      super(_PF, self).__init__()
      self.pv_conv = nn.Sequential(
          nn.Conv2d(2, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Conv2d(dim_pv // reduction, dim_pv, 1, padding=0, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )

      self.art_conv = nn.Sequential(
          nn.Conv2d(2, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Conv2d(dim_pv // reduction, dim_pv, 1, padding=0, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )
      self.Fagg_conv = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction),
          nn.ReLU(),
          nn.Dropout(0.1),
          nn.Conv2d(dim_pv // reduction, dim_pv, 1, padding=0, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )

      self.local_conv_1 = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 3, padding=1, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.local_conv_2 = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 5, padding=2, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.global_conv = nn.Sequential(
          nn.Conv2d(4, dim_pv // reduction, 7, padding=3, bias=False),
          nn.BatchNorm2d(dim_pv // reduction)
      )
      self.gap = _AsppPooling(4, dim_pv // reduction, nn.BatchNorm2d, norm_kwargs=None)
      
      self.fuse = nn.Sequential(
          nn.Conv2d(4 * dim_pv // reduction, dim_pv, 1, padding=0, bias=False),
          nn.BatchNorm2d(dim_pv),
          nn.Sigmoid()
      )
      self.softmax = nn.Softmax(dim=1)

  def forward(self, pv, art):
      pv_avg = torch.mean(pv, dim=1, keepdim=True)
      pv_max, _ = torch.max(pv, dim=1, keepdim=True)
      art_avg = torch.mean(art, dim=1, keepdim=True)
      art_max, _ = torch.max(art, dim=1, keepdim=True)

      feature_concat1 = torch.cat((pv_avg, pv_max), dim=1)
      feature_concat2 = torch.cat((art_avg, art_max), dim=1)
      feature_concat = torch.cat((pv_avg, pv_max, art_avg, art_max), dim=1)

      Fagg_weight = torch.cat((self.local_conv_1(feature_concat), self.local_conv_2(feature_concat),
                             self.global_conv(feature_concat), self.gap(feature_concat)), dim=1)
      Fagg_weight = self.fuse(Fagg_weight).unsqueeze(1)

      pv_weight = self.pv_conv(feature_concat1).unsqueeze(1)
      art_weight = self.art_conv(feature_concat2).unsqueeze(1)

      weights = self.softmax(torch.cat((pv_weight, art_weight,Fagg_weight), dim=1))
      pv_weight, art_weight, Fagg_weight = weights[:, 0:1, :, :, :].squeeze(1), weights[:, 1:2, :, :, :].squeeze(1),weights[:, 2:3, :, :, :].squeeze(1)

      aggregated_feature = pv.mul(pv_weight+Fagg_weight)+art.mul(art_weight+Fagg_weight)
      modulated_pv = (pv+aggregated_feature) / 2
      modulated_art = (art+aggregated_feature) / 2

      return modulated_pv, modulated_art, aggregated_feature

class _FI(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()

        self.pv_conv1_1 = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        self.pv_conv1_2 = nn.Conv2d(skip_ch, out_ch // 2, 1, padding=0, bias=False)
        self.art_conv1_1 = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        self.art_conv1_2 = nn.Conv2d(skip_ch, out_ch // 2, 1, padding=0, bias=False)
        self.pv_classifier = nn.Conv2d(out_ch, 1, 1, padding=0, bias=False)
        self.art_classifier = nn.Conv2d(out_ch, 1, 1, padding=0, bias=False)

        self.pv_conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, aggr, pv_skip, art_skip, mask):
        pv_merge = self.pv_conv1_1(aggr)
        pv_feature = self.pv_conv3(pv_merge)
        pv_pred = self.pv_classifier(pv_feature)

        pv_skip = self.pv_conv1_2(pv_skip)
        art_skip = self.art_conv1_2(art_skip)

        # binary prediction
        if self.training:
            mask_W = pv_pred.size()[3]
            art_pred_mask = F.adaptive_max_pool2d(mask, output_size=mask_W)
        else:
            art_pred_mask = pv_pred > 0

        # masked average pooling
        art_skip_masked = art_skip * art_pred_mask

        B, C, H, W = art_skip_masked.size()

        art_skip_masked = art_skip_masked.view(B, C, -1)

        art_pred_mask = art_pred_mask.view(B, -1)

        art_vector = torch.sum(art_skip_masked, dim=2) / (1e-4 + torch.sum(art_pred_mask, dim=1)).unsqueeze(dim=1).expand([B, C])

        art_vector = art_vector.unsqueeze(dim=2).unsqueeze(dim=3).expand([B, C, H, W])


        pv_similarity = F.cosine_similarity(pv_skip, art_vector, dim=1).unsqueeze(dim=1)

        pv_feature = (pv_feature * pv_similarity) + pv_feature

        return pv_feature, pv_pred

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnext50_32x4d(pretrained=True)
        self.block_1 = nn.Sequential(*list(self.base.children())[:4])
        self.block_2 = nn.Sequential(*list(self.base.children())[4:5])
        self.block_3 = nn.Sequential(*list(self.base.children())[5:6])
        self.block_4 = nn.Sequential(*list(self.base.children())[6:7])
        self.block_5 = nn.Sequential(*list(self.base.children())[7:8])

        self.PF_1 = _PF(64 * 4)
        self.PF_2 = _PF(128 * 4)
        self.PF_3 = _PF(256 * 4)
        self.PF_4 = _PF(512 * 4)


    def forward(self, x_pv, x_art):
        pv_scale_1 = self.block_1(x_pv)
        art_scale_1 = self.block_1(x_art)

        pv_scale_2 = self.block_2(pv_scale_1)
        art_scale_2 = self.block_2(art_scale_1)
        pv_scale_2, art_scale_2, aggr_scale_2 = self.PF_1(pv_scale_2, art_scale_2)

        pv_scale_3 = self.block_3(pv_scale_2)
        art_scale_3 = self.block_3(art_scale_2)
        pv_scale_3, art_scale_3, aggr_scale_3 = self.PF_2(pv_scale_3, art_scale_3)

        pv_scale_4 = self.block_4(pv_scale_3)
        art_scale_4 = self.block_4(art_scale_3)
        pv_scale_4, art_scale_4, aggr_scale_4 = self.PF_3(pv_scale_4, art_scale_4)

        pv_scale_5 = self.block_5(pv_scale_4)
        art_scale_5 = self.block_5(art_scale_4)
        pv_scale_5, art_scale_5, aggr_scale_5 = self.PF_4(pv_scale_5, art_scale_5)

        aggr_scale_tuple = (aggr_scale_2, aggr_scale_3, aggr_scale_4, aggr_scale_5)
        pv_scale_tuple = (pv_scale_2, pv_scale_3, pv_scale_4, pv_scale_5)
        art_scale_tuple = (art_scale_2, art_scale_3, art_scale_4, art_scale_5)

        return aggr_scale_tuple, pv_scale_tuple, art_scale_tuple


class Seg_head(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.aspp = _ASPP(512*4, [6, 12, 18], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        # self.conv_block_1 = _ConvBNReLU(128*block.expansion, 256, 3, padding=1)
        self.conv_5 = _ConvBNReLU(256, 256, 3, padding=1)
        self.conv_4 = _ConvBNReLU(256 * 4, 256, 3, padding=1)
        self.conv_3 = _ConvBNReLU(128 * 4, 256, 3, padding=1)
        self.conv_2 = _ConvBNReLU(64 * 4, 256, 3, padding=1)

        self.conv_block = nn.Sequential(
            _ConvBNReLU(256 * 4, 256, 3, padding=1),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            _ConvBNReLU(256, 256, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        # fusion-block
        self.FI5 = _FI(256, 256, 2048)
        self.FI4 = _FI(1024, 1024, 1024)
        self.FI3 = _FI(512, 512, 512)
        self.FI2 = _FI(256, 256, 256)

    def forward(self, aggr_scale_tuple, pv_scale_tuple, art_scale_tuple, mask):

        aggr_2, aggr_3, aggr_4, aggr_5 = aggr_scale_tuple
        size = aggr_2.size()[2:]
        aggr_5 = self.aspp(aggr_5)
        aggr_5, aggr_5_pred = self.FI5(aggr_5, pv_scale_tuple[3], art_scale_tuple[3], mask)

        aggr_5 = F.interpolate(aggr_5, size, mode='bilinear', align_corners=True)
        aggr_5_pred = F.interpolate(aggr_5_pred, size, mode='bilinear', align_corners=True)
        aggr_5 = self.conv_5(aggr_5)

        aggr_4, aggr_4_pred = self.FI4(aggr_4, pv_scale_tuple[2], art_scale_tuple[2], mask)
        aggr_4 = F.interpolate(aggr_4, size, mode='bilinear', align_corners=True)
        aggr_4_pred = F.interpolate(aggr_4_pred, size, mode='bilinear', align_corners=True)
        aggr_4 = self.conv_4(aggr_4)

        aggr_3, aggr_3_pred = self.FI3(aggr_3, pv_scale_tuple[1], art_scale_tuple[1], mask)
        aggr_3 = F.interpolate(aggr_3, size, mode='bilinear', align_corners=True)
        aggr_3_pred = F.interpolate(aggr_3_pred, size, mode='bilinear', align_corners=True)
        aggr_3 = self.conv_3(aggr_3)

        aggr_2 = F.interpolate(aggr_2, size, mode='bilinear', align_corners=True)
        aggr_2, aggr_2_pred = self.FI2(aggr_2, pv_scale_tuple[0], art_scale_tuple[0], mask)
        aggr_2 = self.conv_2(aggr_2)

        features = self.conv_block(torch.cat([aggr_5, aggr_4, aggr_3, aggr_2], dim=1))
        maps = self.classifier(features)

        out = (aggr_2_pred, aggr_3_pred, aggr_4_pred, aggr_5_pred)

        return features, maps, out


class Multi_phase_seg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = Encoder()
        self.head = Seg_head(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_pv, x_art, mask):
        size = x_pv.size()[2:]
        aggr_scale_tuple, pv_scale_tuple, art_scale_tuple = self.encoder(x_pv, x_art)

        features, maps, out = self.head(aggr_scale_tuple, pv_scale_tuple, art_scale_tuple, mask)
        final_seg = F.interpolate(maps, size, mode='bilinear', align_corners=True)

        return out, final_seg


if __name__ == '__main__':
    USE_GPU = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_GPU else "cpu")

    a = torch.ones((2, 3, 256, 256)).to(device)
    b = torch.ones((2, 3, 256, 256)).to(device)
    c = torch.ones((2, 1, 256, 256)).to(device)

    model = Multi_phase_seg(num_classes=2).to(device)
    out, predicts = model(a,b,c)