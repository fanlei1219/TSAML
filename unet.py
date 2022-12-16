import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy, deepcopy
import warnings

warnings.filterwarnings('ignore')


class UNeT(nn.Module):
    def __init__(self):
        super(UNeT, self).__init__()
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        # inc DoubleConv(1, 64)
        weight = nn.Parameter(torch.ones(64, 3, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # down1 Down(64, 128)
        weight = nn.Parameter(torch.ones(128, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(128))
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(128), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(128), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(128, 128, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(128))
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(128), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(128), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # down2 Down(128, 256)
        weight = nn.Parameter(torch.ones(256, 128, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(256, 256, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # down3 Down(256, 512)
        weight = nn.Parameter(torch.ones(512, 256, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(512))
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(512), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(512), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(512, 512, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(512))
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(512), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(512), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # down4 Down(512, 512)
        weight = nn.Parameter(torch.ones(512, 512, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(512))
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(512), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(512), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(512, 512, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(512))
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(512), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(512), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # up1 Up(in=1024, out=256, mid=512)
        weight = nn.Parameter(torch.ones(512, 1024, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(512))
        bias = nn.Parameter(torch.zeros(512))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(512), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(512), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(256, 512, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # up2 Up(in=512, out=128, mid=256)
        weight = nn.Parameter(torch.ones(256, 512, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(128, 256, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(128))
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(128), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(128), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # up3 Up(in=256, out=64, mid=128)
        weight = nn.Parameter(torch.ones(128, 256, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(128))
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(128), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(128), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(64, 128, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # up4 Up(128, 64, 64)
        weight = nn.Parameter(torch.ones(64, 128, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
        weight = nn.Parameter(torch.ones(64, 64, 3, 3))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])
        running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # outc OutConv(64, 1)
        weight = nn.Parameter(torch.ones(1, 64, 1, 1))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(1))
        self.vars.extend([weight, bias])

    def forward(self, x, params=None, bn_training=True):
        if params is None:
            params = self.vars

        # inc(x)
        weight, bias = params[0], params[1]
        x1 = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[2], params[3]
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x1 = F.batch_norm(x1, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x1 = F.relu(x1, inplace=True)
        weight, bias = params[4], params[5]
        x1 = F.conv2d(x1, weight, bias, stride=1, padding=1)
        weight, bias = params[6], params[7]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x1 = F.batch_norm(x1, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x1 = F.relu(x1, inplace=True)
        # print(x1.shape)

        # down1(x)
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        weight, bias = params[8], params[9]
        x2 = F.conv2d(x2, weight, bias, stride=1, padding=1)
        weight, bias = params[10], params[11]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        x2 = F.batch_norm(x2, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x2 = F.relu(x2, inplace=True)
        weight, bias = params[12], params[13]
        x2 = F.conv2d(x2, weight, bias, stride=1, padding=1)
        weight, bias = params[14], params[15]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        x2 = F.batch_norm(x2, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x2 = F.relu(x2, inplace=True)
        # print(x2.shape)

        # down2(x)
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        weight, bias = params[16], params[17]
        x3 = F.conv2d(x3, weight, bias, stride=1, padding=1)
        weight, bias = params[18], params[19]
        running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
        x3 = F.batch_norm(x3, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x3 = F.relu(x3, inplace=True)
        weight, bias = params[20], params[21]
        x3 = F.conv2d(x3, weight, bias, stride=1, padding=1)
        weight, bias = params[22], params[23]
        running_mean, running_var = self.vars_bn[10], self.vars_bn[11]
        x3 = F.batch_norm(x3, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x3 = F.relu(x3, inplace=True)
        # print(x3.shape)

        # down3(x)
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        weight, bias = params[24], params[25]
        x4 = F.conv2d(x4, weight, bias, stride=1, padding=1)
        weight, bias = params[26], params[27]
        running_mean, running_var = self.vars_bn[12], self.vars_bn[13]
        x4 = F.batch_norm(x4, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x4 = F.relu(x4, inplace=True)
        weight, bias = params[28], params[29]
        x4 = F.conv2d(x4, weight, bias, stride=1, padding=1)
        weight, bias = params[30], params[31]
        running_mean, running_var = self.vars_bn[14], self.vars_bn[15]
        x4 = F.batch_norm(x4, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x4 = F.relu(x4, inplace=True)
        # print(x4.shape)

        # down4(x)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        weight, bias = params[32], params[33]
        x5 = F.conv2d(x5, weight, bias, stride=1, padding=1)
        weight, bias = params[34], params[35]
        running_mean, running_var = self.vars_bn[16], self.vars_bn[17]
        x5 = F.batch_norm(x5, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x5 = F.relu(x5, inplace=True)
        weight, bias = params[36], params[37]
        x5 = F.conv2d(x5, weight, bias, stride=1, padding=1)
        weight, bias = params[38], params[39]
        running_mean, running_var = self.vars_bn[18], self.vars_bn[19]
        x5 = F.batch_norm(x5, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x5 = F.relu(x5, inplace=True)

        low_feature = x5
        # print(x5.shape)

        # up1(x5,x4)
        x5 = F.upsample_bilinear(x5, scale_factor=2)
        diffY1 = x4.size()[2] - x5.size()[2]
        diffX1 = x4.size()[3] - x5.size()[3]
        x5 = F.pad(x5, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])
        x = torch.cat([x4, x5], dim=1)
        weight, bias = params[40], params[41]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[42], params[43]
        running_mean, running_var = self.vars_bn[20], self.vars_bn[21]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        weight, bias = params[44], params[45]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[46], params[47]
        running_mean, running_var = self.vars_bn[22], self.vars_bn[23]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        # print(x.shape)

        # up2(x, x3)
        x = F.upsample_bilinear(x, scale_factor=2)
        diffY2 = x3.size()[2] - x.size()[2]
        diffX2 = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX2 // 2, diffX2 - diffX2 // 2,
                      diffY2 // 2, diffY2 - diffY2 // 2])
        x = torch.cat([x3, x], dim=1)
        weight, bias = params[48], params[49]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[50], params[51]
        running_mean, running_var = self.vars_bn[24], self.vars_bn[25]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        weight, bias = params[52], params[53]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[54], params[55]
        running_mean, running_var = self.vars_bn[26], self.vars_bn[27]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        # print(x.shape)

        # up3(x, x2)
        x = F.upsample_bilinear(x, scale_factor=2)
        diffY3 = x2.size()[2] - x.size()[2]
        diffX3 = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX3 // 2, diffX3 - diffX3 // 2,
                      diffY3 // 2, diffY3 - diffY3 // 2])
        x = torch.cat([x2, x], dim=1)
        weight, bias = params[56], params[57]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[58], params[59]
        running_mean, running_var = self.vars_bn[28], self.vars_bn[29]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        weight, bias = params[60], params[61]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[62], params[63]
        running_mean, running_var = self.vars_bn[30], self.vars_bn[31]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        # print(x.shape)

        # up4(x, x1)
        x = F.upsample_bilinear(x, scale_factor=2)
        diffY4 = x1.size()[2] - x.size()[2]
        diffX4 = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX4 // 2, diffX4 - diffX4 // 2,
                      diffY4 // 2, diffY4 - diffY4 // 2])
        x = torch.cat([x1, x], dim=1)
        weight, bias = params[64], params[65]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[66], params[67]
        running_mean, running_var = self.vars_bn[32], self.vars_bn[33]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        weight, bias = params[68], params[69]
        x = F.conv2d(x, weight, bias, stride=1, padding=1)
        weight, bias = params[70], params[71]
        running_mean, running_var = self.vars_bn[34], self.vars_bn[35]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)
        x = F.relu(x, inplace=True)
        # print(x.shape)

        # outc(x)
        weight, bias = params[72], params[73]
        x = F.conv2d(x, weight, bias, stride=1)
        # print(x.shape)

        output = x
        return output ,low_feature

    def parameters(self):
        return self.vars


if __name__ == "__main__":
    x = torch.rand(5, 3, 512, 512)
    net = UNeT()
    output,low_feature = net(x, params=None, bn_training=True)
    print(output.shape)
    print(low_feature.shape)
