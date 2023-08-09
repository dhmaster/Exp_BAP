import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
EPSILON = 1e-12

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

# Bilinear Attention Pooling
class BAP1(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP1, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            # attentions = F.upsample_bilinear(attentions, size=(H, W))
            attentions = nn.functional.interpolate(attentions, size=(H, W), mode='bilinear', align_corners=False)

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = []
            for i in range(M):
                AiF = (features * attentions[:, i:i + 1, ...])
                AiF = torch.max(AiF, dim=1)[0]
                AiF = torch.unsqueeze(AiF, dim=1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)
            # feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) +  EPSILON)

        # l2 normalization along dimension M and C

        feature_matrix = F.normalize(feature_matrix, dim=-1)
        # print(feature_matrix.shape)
        return feature_matrix

class ResizeCat(nn.Module):
    def __init__(self,**kwargs):
        super(ResizeCat, self).__init__()
        pass
    def forward(self, fea_maps, att_maps):
        N, C, H, W = fea_maps.size()
        if C != 512:
            conv = nn.Conv2d(C, 512, kernel_size=1)
            fea_maps = conv(fea_maps)
        resized_at3 = nn.functional.interpolate(att_maps, (H, W))  # 实现插值和上采样，前两维不会变
        # print(resized_at3.size())
        cat_fea = torch.cat((fea_maps, resized_at3), dim=1)
        return cat_fea

##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max() # 从theta指定的范围内任取一个数乘于atten_map的最大值
            else:
                theta_d = theta * atten_map.max()
            atten_map_upsampled = torch.nn.functional.interpolate(atten_map, size=(imgH, imgW), mode='bilinear',
                                                                  align_corners=False)
            drop_masks.append(atten_map_upsampled < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images
    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

if __name__ == '__main__':
    a = BAP1()
    # a = ResizeCat()
    a1 = torch.Tensor(12, 16, 26, 26)
    a3 = torch.Tensor(12, 5, 26, 26)
    a5 = torch.Tensor(4, 9, 9, 9)
    # ret = a(a1,a3,a5)
    ret = a(a1, a3)
    # print(ret[0].size(),ret[1].size())



