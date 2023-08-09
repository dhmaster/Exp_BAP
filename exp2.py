from __future__ import print_function
import math
import os
import time
import warnings
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchinfo import summary
from torchvision.transforms import transforms
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from loss import AverageMeter,TopKAccuracyMetric
import config
from torchstat import stat

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5,momentum=0.01,affine=True) if bn else None
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

# num_features:CNN输出值
# M：M个attentions map
# attentions = BasicConv2d(num_features, M, kernel_size=1)
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# self.spp = SPPLayer(pool_size=[1, 2, 4])
# self.fc = nn.Linear(512 * block.expansion * self.spp.out_length, num_classes)
class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool((h_wid, w_wid), stride=(h_wid, w_wid), padding=int(h_pad))(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp

class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class):
        super().__init__()
        self.Features = Features(net_layers)
        self.att = nn.Sequential(
            BasicConv2d(2048, 512, kernel_size=1)
        )
        self.att1 = nn.Sequential(
            BasicConv2d(1024, 512, kernel_size=1)
        )
        self.upsample = ResizeCat()
        self.bap = BAP1()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=True),
            nn.Linear(2048, num_class)
        )
        self.spp = SPPLayer(pool_size=[2])

    def forward(self, x):
        batch_size = x.size(0)
        x1, x2, x3 = self.Features(x)
        att3 = self.att(x3)
        fea3 = self.bap(x3, att3)       # N*512*14*14
        in_2 = self.upsample(x2, fea3)  # N*1024*28*28

        att2 = self.att1(in_2)
        fea2 = self.bap(in_2, att2)     # N*512*28*28
        in_1 = self.upsample(x1, fea2)  # N*1024*56*56

        att1 = self.att1(in_1)
        fea1 = self.bap(in_1, att1)     # N*512*56*56

        fc = self.spp(fea1)
        fc = torch.flatten(fc, 1)       # N * 2048
        pre_raw = self.classifier(fc)   # N * num_class
        print(fc.shape, pre_raw.shape)

        attention_maps = fea1[:, :32, :, :]
        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(32, 1, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        return pre_raw,attention_map



def train(net,device,epoch,dataloader,optimizer):
        start_time = time.time()
        net.train()
        print('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        # metrics initialization
        loss_container.reset()
        raw_metric.reset()
        drop_metric.reset()
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            idx = batch_idx  # 6667/16 = 417 idx: 0 - 416
            if inputs.shape[0] < config.batch_size:
                continue
            if config.use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, config.num_epoch, config.lr[nlr])
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
            ##################################
            # Raw Image
            ##################################
            # raw images forward
            y_pred_raw, attention_map = net(inputs)
            ##################################
            # Attention Dropping
            ##################################
            with torch.no_grad():
                drop_images = batch_augment(inputs, attention_map, mode='drop', theta=(0.2, 0.5))
            # drop images forward
            y_pred_drop, _ = net(drop_images)
            # loss
            batch_loss = CELoss(y_pred_raw, targets)*0.6 + CELoss(y_pred_drop, targets)*0.4

            # backward
            batch_loss.backward()
            optimizer.step()

            # metrics: loss and top-1,5 error
            with torch.no_grad():
                epoch_loss = loss_container(batch_loss.item())
                epoch_raw_acc = raw_metric(y_pred_raw, targets)
                epoch_drop_acc = drop_metric(y_pred_drop, targets)
            # end of this batch
            if batch_idx % 50 == 0:
                batch_info = ('Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
                    epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1], epoch_drop_acc[0], epoch_drop_acc[1]))
                # print('Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
                #     epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1], epoch_drop_acc[0], epoch_drop_acc[1]))
        # end of this epoch
        end_time = time.time()
        with open(config.store_name + '/results_train.txt', 'a') as file:
           file.write('Iteration {}: Learning Rate {:g},{:g} Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f}, Time {:3.2f})\n'.format(
                    epoch+1,optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],epoch_loss,epoch_raw_acc[0],epoch_raw_acc[1],epoch_drop_acc[0],epoch_drop_acc[1],end_time-start_time))

def test(net,device,dataloader):
    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    # begin validation
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if config.use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            y_pre_raw, attention_map = net(inputs)
            ##################################
            # Object Localization and Refinement
            ##################################
            crop_images = batch_augment(inputs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _ = net(crop_images)
            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pre_raw + y_pred_crop) / 2.

            # loss
            batch_loss = CELoss(y_pred, targets)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, targets)
        if batch_idx % 50 == 0:
            print('Step:{} Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(
                batch_idx,epoch_loss,epoch_acc[0],epoch_acc[1]))
    # end of validation
    return epoch_loss, epoch_acc

CELoss = nn.CrossEntropyLoss()
# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

def main():
    exp_dir = config.store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    logs = {}
    max_val_acc = 0
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=config.data_path + '/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)  # 4
    testset = torchvision.datasets.ImageFolder(root=config.data_path + '/test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)  #

    # Instantiate model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = torchvision.models.resnet50()
    # 加载你的权重文件
    net.load_state_dict(torch.load(config.model_weight_path))
    net_layers = list(net.children())  # 返回最外层元素
    net_layers = net_layers[0:8]
    net = Network_Wrapper(net_layers, 100)
    net.to(device)

    optimizer = optim.SGD([
        {'params': net.att.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}
    ], momentum=0.9, weight_decay=5e-4)

    start_epoch = 0
    ##################################
    # Logging setting
    ##################################
    logs = {}
    # 如果文件夹路径不存在，则创建文件夹
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
    ##################################
    # checkpiont 设置恢复之前的训练状态
    ##################################
    if config.RESUME: #ckpt
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.pth_checkpoint)
        # Load weights
        start_epoch = checkpoint['start_epoch']
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    logging.info('Network weights save to {}'.format(config.save_dir))

    for epoch in range(start_epoch, config.num_epoch):
        train(net,device,epoch,trainloader,optimizer)
        if epoch >= 5:
            epoch_loss, epoch_acc = test(net,device,testloader)
            if epoch_acc[0] > max_val_acc:
                max_val_acc = epoch_acc[0]
                checkpoint = {
                    'start_epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
            torch.save(checkpoint, 'checkpoint' + '/ckpt_best_%s.pth' % (str(epoch)))

            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration {}, Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})\n'.format(
                    epoch,epoch_loss,epoch_acc[0],epoch_acc[1]))
        else:
            torch.save(checkpoint, 'checkpoint' + '/ckpt_best_%s.pth' % (str(epoch)))

if __name__ == '__main__':
    main()
    # net = torchvision.models.resnet50()
    # 加载你的权重文件
    # model_weight_path = "./resnet50-19c8e357.pth"  # 将此路径更改为你的权重文件路径
    # net.load_state_dict(torch.load(model_weight_path))
    # net_layers = list(net.children())  # 返回最外层元素
    # net_layers = net_layers[0:8]
    # net = Network_Wrapper(net_layers, 100)
    #
    # stat(net, (3, 224, 224))
    # summary(net,(2,3,448,448))
    # net = ResNet50()
    # print(net.Features.net_layer_4[0].conv2.weight.data)

    # a = torch.rand([4, 3, 448, 448])
    # output_2 = net(a)
    # print(output_2.shape)
    # for param in net.state_dict():
    #     print(param, "\t", net.state_dict()[param].size())




