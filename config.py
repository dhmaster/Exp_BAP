##################################################
# File Record
##################################################
store_name = 'Results_FGVC_Aircraft_ResNet50_1'
data_path = './FGVC_Aircraft'
# log
save_dir = './ckpt/'
log_name = 'train.log'
# checkpoint
pth_checkpoint = './checkpoint/ckpt_best_19.pth' # 断点
##################################################
# Training Config
##################################################
# GPU = '0'                   # GPU
RESUME = False               # load weight form where trainning stop
workers = 0                 # number of Dataloader workers
num_epoch = 160              # total number of epochs
batch_size = 12             # batch size
# learning_rate = 1e-3        # initial learning rate
use_cuda = True
lr = [0.002, 0.0002]


##################################################
# Model Config
##################################################
model_weight_path = "./resnet50-19c8e357.pth"  # 加载预训练权重
image_size = (448, 448)     # size of training images
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 32         # number of attention maps
beta = 5e-2                 # param for update feature centers


##################################################
# Dataset/Path Config
##################################################
tag = 'bird'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models

model_name = 'model.ckpt'


# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = True
eval_ckpt = save_dir + model_name
eval_savepath = './FGVC/CUB-200-2011/visualize/'