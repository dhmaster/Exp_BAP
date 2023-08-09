import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# visualize
visualize = True
savepath = './heatmap'
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(2048,1,kernel_size=1)
    def forward(self,x):
        return self.conv(x)

def main():
    device = torch.device("cuda")
    net = torchvision.models.resnet50()
    # 加载预训练权重
    model_weight_path = "./resnet50-19c8e357.pth"  # 修改为你的权重文件路径
    pretrained_dict = torch.load(model_weight_path)
    model_dict = net.state_dict()

    # 剔除全连接层和池化层的权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       not k.startswith('fc') and not k.startswith('avgpool')}

    # 加载预训练权重到模型
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    # 去除最后的池化层和全连接层
    modules = list(net.children())[:-2]
    net = nn.Sequential(*modules)
    net.to(device)
    net.eval()
    conv = Conv()
    conv.cuda()
    transform_test = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root='./FGVC_Aircraft/test',
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=0)  #
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')

        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            attention_maps = net(X)
            attention_maps.cuda()
            attention_maps = conv(attention_maps)
            # # Augmentation with crop_mask
            # crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)
            #
            # y_pred_crop, _, _ = net(crop_image)
            # y_pred = (y_pred_raw + y_pred_crop) / 2.

            if visualize:
                # reshape attention maps
                attention_maps = F.interpolate(attention_maps, size=(X.size(2), X.size(3)), mode='bilinear',
                                               align_corners=False)
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i)))

            # Top K
            # epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            # epoch_ref_acc = ref_accuracy(y_pred, y)

            # end of this batch
            print('END')
            pbar.update()
        pbar.close()

if __name__ == '__main__':
    main()