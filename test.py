import argparse
import os

import cv2
import numpy as np
import torch
import torchvision
from torch.autograd import Function
from torchvision import models
from exp2 import Network_Wrapper
""" 
Class for extracting activations and
registering gradients from targetted intermediate layers 
"""
class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    # 钩子
    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():  # 遍历model的每一个模块，比如卷积、BN、ReLU
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)  # 利用hook来记录目标层的梯度
                outputs += [x]
        return outputs, x                            # output保留目标层的输出和梯度


class ModelOutputs():
    """
    Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers.
    """
    # feature_module是我们想要提取特征的模块，target_layers是我们想要在feature_module中获取到输出并保存梯度的层的名字列表
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
    # 用于返回梯度的方法
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            # 当到达指定输出的网络层时，输出该层的激活特征值并保留该层的输出
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            # 如果模块名称中包含"avgpool"（这通常指的是平均池化层），我们会手动进行前向传播并改变张量的形状
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        # 返回保存的目标层的激活和最后的模型输出
        return target_activations, x
'''
 图像预处理函数
'''
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # 拷贝并且翻转其颜色通道BGR->RGB. 在OpenCV库中，图像默认是以BGR方式读取的
    preprocessed_img = img.copy()[:, :, ::-1]
    # 对每个颜色通道分别执行标准化操作
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    # 将颜色通道从最后一个维度移动到第一个维度，因为PyTorch处理的图像是以(C, H, W)的方式组织的
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    # transpose HWC > CHW
    # 将numpy数组转换为Pytorch的tensor
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

'''
 将一个类激活映射(CAM)可视化，把它叠加到原始图像上
'''
def show_cam_on_image(img, mask):
    # 首先把mask矩阵缩放到0-255即像素值的正常范围内（np.uint8(255 * mask)）。然后，使用cv2.applyColorMap函数，将这个单通道的灰度图映射成一个色彩丰富的heat map
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # 将归一化后的heatmap和原图像进行相加，img图像的值也在0-1范围内，如果不是，需要先进行归一化
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # np.uint8(255 * cam): 将归一化后的cam图像缩放回0-255区间
    cv2.imwrite("./checkpoint/cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
# 得到模型最后一层输出特征的分类概率,根据输出的概率的大小判断它与输入图片中哪些区域的像素值更敏感

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()  # 梯度清零
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)  # 反向传播取得目标层梯度

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # 调用函数get_gradients(),  得到目标层求得的梯度（目标层特征图的每个像素的偏导数）
        target = features[-1]  # feature 包含卷积、BN、和ReLU层，这里取ReLU作为最后的特征值

        target = target.cpu().data.numpy()[0, :]
        # 忽略Batch数目，只取C,H,W维度
        print(target.shape)

        # 特征图每个像素的偏导数求出来之后，取一次宽高维度上的全局平均，得到c类相对于该卷积层输出特征图的第k个通道的敏感程度 weifhts.shape=[n,c]
        weights = np.mean(grads_val, axis=(2, 3))[0, :]

        print(weights.shape)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        # target[i, :, :] = array:shape(H, W)累加各个通道的特征值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        print(cam.shape)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str,
                        default='./1.jpg',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


"""
https://blog.csdn.net/gusui7202/article/details/83239142
qhy。
"""

import os



if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    data_dir = "./FGVC_Aircraft/test"
    file_list = []
    write_file_name = './test.txt'
    write_file = open(write_file_name, "w")
    tt = os.listdir(data_dir)
    for classes in tt:
        imgs = os.listdir(os.path.join(data_dir,classes))
        # for imgs in classes:
        for i, x in enumerate(imgs):
            write_name = os.path.join(data_dir,classes,x)
            file_list.append(write_name)
    print(file_list)
    for current_line in range(len(file_list)):
        write_file.write(file_list[current_line] + '\n')
    write_file.close()
    # args = get_args()
    #
    # # Can work with any model, but it assumes that the model has a
    # # feature method, and a classifier method,
    # # as in the VGG models in torchvision.
    #
    # # 加载模型及模型的预训练参数
    # model = torchvision.models.resnet50()
    # # 加载你的权重文件
    # model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
    # net_layers = list(model.children())  # 返回最外层元素
    # net_layers = net_layers[0:8]
    # model = Network_Wrapper(net_layers, 100)
    #
    #
    # for name, value in model.classifier._modules.items():
    #     print(" ", name,value)
    # 指定可视化的卷积层，及卷积层中哪个模块的输出
    # grad_cam = GradCam(model=model, feature_module=model.classifier,target_layer_names=["1"], use_cuda=args.use_cuda)
    #
    # # 读取图片归一化为[0,1]矩阵,并转换为张量形式
    # img = cv2.imread(args.image_path, 1)
    # img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    #
    # # If None, returns the map for the highest scoring category.
    # # Otherwise, targets the requested index.
    # target_index = None
    # mask = grad_cam(input, target_index)  # 输出该张图像在指定网络层上，所有卷积通道的特征图叠加后的mask，用于后续和图片的叠加显示
    #
    # show_cam_on_image(img, mask)
    #
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    # gb = gb_model(input, index=target_index)
    # gb = gb.transpose((1, 2, 0))
    # cam_mask = cv2.merge([mask, mask, mask])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    #
    # cv2.imwrite('./checkpoint/gb.jpg', gb)
    # cv2.imwrite('./checkpoint/cam_gb.jpg', cam_gb)