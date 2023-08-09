from __future__ import print_function
import os
from PIL import Image
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.transforms import transforms
from utils import BAP, cosine_anneal_schedule
from utils import *
from torchsummary import summary

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class):
        super().__init__()
        self.Features = Features(net_layers)

        # self.max_pool1 = nn.MaxPool2d(kernel_size=56, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=28, stride=1)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=14, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(16384),
            nn.Linear(16384, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(32768),
            nn.Linear(32768, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(65536),
            nn.Linear(65536, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, num_class),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(114688),
            nn.Linear(114688, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=True),
            nn.Linear(2048, num_class),
        )
        self.bap = BAP()

    def forward(self, x):
        x1, x2, x3 = self.Features(x)  # 2048
        # print(x1.shape, x2.shape, x3.shape)
        x1_ = self.conv_block1(x1)  # stage 3
        attention1 = x1_[:, :32, :, :]
        # print(attention1.shape)
        raw_features1, pooling_features1 = self.bap(x1, attention1)
        x1_f = torch.flatten(pooling_features1, 1)
        # print(x1_f.shape)
        x1_c = self.classifier1(x1_f)
        # print(x1_c.shape)

        x2_ = self.conv_block2(x2)
        attention2 = x2_[:, :32, :, :]
        # print(attention2.shape)
        raw_features2, pooling_features2 = self.bap(x2, attention2)
        x2_f = torch.flatten(pooling_features2, 1)
        # print(x2_f.shape)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        attention3 = x3_[:, :32, :, :]
        # print(attention3.shape)
        raw_features3, pooling_features3 = self.bap(x3, attention3)
        x3_f = torch.flatten(pooling_features3, 1)
        # print(x3_f.shape)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        # print(x_c_all.shape)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, data_path=''):
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)  # 4

    net = torchvision.models.resnet50()
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    net.load_state_dict(state_dict)

    net_layers = list(net.children())  # 返回最外层元素
    net_layers = net_layers[0:8]
    net = Network_Wrapper(net_layers, 100)

    netp = torch.nn.DataParallel(net, device_ids=[0])

    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}
    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        # train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx  # 6667/16 = 417 idx: 0 - 416
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # Train the experts from shallow to deep
            # e1
            optimizer.zero_grad()
            inputs1 = inputs
            output_1, _, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # e2
            optimizer.zero_grad()
            inputs2 = inputs

            _, output_2, _, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # e1
            optimizer.zero_grad()
            flag = torch.rand(1)
            inputs3 = inputs

            _, _, output_3, _ = netp(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()


            # Train the concatenation of the experts with the raw input
            optimizer.zero_grad()
            _, _, _, output_concat = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            # train_loss4 += concat_loss_ATT.item()
            train_loss5 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f |Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss5 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
           file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1),
                    train_loss3 / (idx + 1),
                    train_loss5 / (idx + 1)))

        if epoch < 5 or epoch >= 100:
            val_acc_com, val_loss = test(net, CELoss, 3, data_path + '/test')
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                    epoch, val_acc_com, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)


def test(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat = net(inputs)

        outputs_com2 = output_1 + output_2 + output_3 + output_concat
        # outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        # _, predicted_com = torch.max(outputs_com.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        # correct_com += predicted_com.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1),
                100. * float(correct_com2) / total, correct_com2, total))

    test_acc_en = 100. * float(correct_com2) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss


if __name__ == '__main__':
    net = torchvision.models.resnet50()
    net_layers = list(net.children())  # 返回最外层元素
    net_layers = net_layers[0:8]
    net = Network_Wrapper(net_layers, 100)
    summary(net, (3,448,448))
    # print(net)
    # a = torch.rand([16, 3, 448, 448])
    # _, output_2, _, _ = net(a)
    # for param in net.state_dict():
    #     print(param, "\t", net.state_dict()[param].size())

    # data_path = './FGVC_Aircraft'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # train(nb_epoch=100,  # number of epoch
    #       batch_size=16,  # batch size = 16
    #       store_name='Results_FGVC_Aircraft_ResNet50',  # folder for output
    #       resume=False,  # resume training from checkpoint
    #       start_epoch=0,  # the start epoch number when you resume the training
    #       model_path='',
    #       data_path=data_path)  # the saved model where you want to resume the training


