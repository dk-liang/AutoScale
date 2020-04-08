import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torchvision import models

def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h) / 2.0)):int(math.floor((d_h - g_h) / 2.0)) + g_h,
         int(math.floor((d_w - g_w) / 2.0)):int(math.floor((d_w - g_w) / 2.0)) + g_w]
    return d1


class AutoScale(nn.Module):

    def __init__(self, load_weights=False):
        super(AutoScale, self).__init__()
        self.seen = 0
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=35),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            # conv2
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            # conv3
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv5 = nn.Sequential(
            # conv5
            nn.MaxPool2d(2, stride=1, padding=1, ceil_mode=False),  # 1/8
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 256, 3,padding=2,dilation=2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 128, 3,padding=2,dilation=2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, 64, 3, padding=2,dilation=2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64,11,1)
        )

        self.cd1 = nn.Sequential(nn.Conv2d(64, 24, 3, padding=1),
                                 nn.BatchNorm2d(24))
        self.cd2 = nn.Sequential(nn.Conv2d(128, 24, 3, padding=1),
                                 nn.BatchNorm2d(24))
        self.cd3 = nn.Sequential(nn.Conv2d(256, 24, 3, padding=1),
                                 nn.BatchNorm2d(24))
        self.cd4 = nn.Sequential(nn.Conv2d(512, 24, 3, padding=1),
                                 nn.BatchNorm2d(24))
        self.cd5 = nn.Sequential(nn.Conv2d(512, 24, 3, padding=1),
                                 nn.BatchNorm2d(24))

        self.rd5 = nn.Sequential(nn.Conv2d(24, 8, 1),
                                 nn.BatchNorm2d(8))
        self.rd4 = nn.Sequential(nn.Conv2d(32, 8, 1),
                                 nn.BatchNorm2d(8))
        self.rd3 = nn.Sequential(nn.Conv2d(32, 8, 1),
                                 nn.BatchNorm2d(8))
        self.rd2 = nn.Sequential(nn.Conv2d(32, 8, 1),
                                 nn.BatchNorm2d(8))

        self.up5 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up4 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up3 = nn.ConvTranspose2d(8, 8, 4, stride=2)
        self.up2 = nn.ConvTranspose2d(8, 8, 4, stride=2)

        self.dsn1 = nn.Conv2d(32, 11, 1)
        self.dsn2 = nn.Conv2d(32, 11, 1)
        self.dsn3 = nn.Conv2d(32, 11, 1)
        self.dsn4 = nn.Conv2d(32, 11, 1)
        self.dsn5 = nn.Conv2d(24, 11, 1)
        self.dsn6 = nn.Conv2d(55, 11, 1)

        self.upscore2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upscore3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upscore4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.upscore5 = nn.UpsamplingBilinear2d(scale_factor=8)

        # self.conv1_1 = nn.Sequential(
        #     nn.Conv2d(32, 32, 1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv1_2 = nn.Sequential(
        #     nn.Conv2d(32, 32, 1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv1_3 = nn.Sequential(
        #     nn.Conv2d(32, 32, 1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv1_4 = nn.Sequential(
        #     nn.Conv2d(32, 32, 1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv1_5 = nn.Sequential(
        #     nn.Conv2d(24, 24, 1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.conv1_6 = nn.Sequential(
        #     nn.Conv2d(152, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        # )



        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()

            # self.conv1.state_dict().items()[0][1].data[:] = mod.state_dict().items()[0][1].data[:]
            # self.conv2.state_dict().items()[1][1].data[:] = mod.state_dict().items()[1][1].data[:]
            # self.conv3.state_dict().items()[2][1].data[:] = mod.state_dict().items()[2][1].data[:]
            # self.conv4.state_dict().items()[3][1].data[:] = mod.state_dict().items()[3][1].data[:]
            # self.conv5.state_dict().items()[4][1].data[:] = mod.state_dict().items()[4][1].data[:]
            #
            #
            # for i in xrange(len(self.frontend.state_dict().items())):
            #     self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, gt, refine_flag ):
        # return h
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        gt = torch.unsqueeze(gt, 1)

        p5 = self.cd5(conv5)
        d5 = self.upscore5(self.dsn5(F.relu(p5)))
        d5 = crop(d5, gt)

        p5_up = self.rd5(F.relu(p5))
        p4_1 = self.cd4(conv4)
        p4_2 = crop(p5_up, p4_1)
        p4_3 = F.relu(torch.cat((p4_1, p4_2), 1))
        p4 = p4_3
        d4 = self.upscore4(self.dsn4(p4))
        d4 = crop(d4, gt)

        p4_up = self.up4(self.rd4(F.relu(p4)))
        p3_1 = self.cd3(conv3)
        p3_2 = crop(p4_up, p3_1)
        p3_3 = F.relu(torch.cat((p3_1, p3_2), 1))
        p3 = p3_3
        d3 = self.upscore3(self.dsn3(p3))
        d3 = crop(d3, gt)

        p3_up = self.up3(self.rd3(F.relu(p3)))
        p2_1 = self.cd2(conv2)
        p2_2 = crop(p3_up, p2_1)
        p2_3 = F.relu(torch.cat((p2_1, p2_2), 1))
        p2 = p2_3
        d2 = self.upscore2(self.dsn2(p2))
        d2 = crop(d2, gt)

        p2_up = self.up2(self.rd2(F.relu(p2)))
        p1_1 = self.cd1(conv1)
        p1_2 = crop(p2_up, p1_1)
        p1_3 = F.relu(torch.cat((p1_1, p1_2), 1))
        p1 = p1_3
        d1 = self.dsn1(p1)
        d1 = crop(d1, gt)

        d6 = self.dsn6(torch.cat((d1, d2, d3, d4, d5), 1))

        if refine_flag ==True:
            p1=F.upsample_bilinear(p1, (d6.size()[2], d6.size()[3]))
            #p1 =self.conv1_1(p1)

            p2=F.upsample_bilinear(p2, (d6.size()[2], d6.size()[3]))
            #p2 = self.conv1_2(p2)

            p3=F.upsample_bilinear(p3, (d6.size()[2], d6.size()[3]))
            #p3 = self.conv1_3(p3)

            p4=F.upsample_bilinear(p4, (d6.size()[2], d6.size()[3]))
            #p4 = self.conv1_4(p4)

            p5=F.upsample_bilinear(p5, (d6.size()[2], d6.size()[3]))
            #p5 = self.conv1_5(p5)

            scale_extract = torch.cat((p1, p2, p3, p4, p5), 1)
            #scale_extract = self.conv1_6(scale_extract)

            return d1, d2, d3, d4, d5, d6, scale_extract

        else:
            return d1, d2, d3, d4, d5, d6