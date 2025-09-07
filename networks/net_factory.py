from networks.unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3,UNet_URPC
from networks.deeplabv2 import get_deeplab_v2
from networks.deeplabv2_ours import get_deeplab_v2_ours
# from networks.sifa_model import Encoder
# from nets.segformer_maal import SegFormer
# from networks.deeplab_dclps import Deeplab
# from networks.deeplab_vgg import DeeplabVGG
# from networks.deeplabv3 import DeepLab
# from deeplabv3_mobv3.deeplabv3_model import create_model,create_model_ours
# from deeplabv2152.deeplbv2_resnet151 import build_feature_extractor
# from unet import UNet, MCNet2d_v1, MCNet2d_v2, MCNet2d_v3,UNet_URPC
# from deeplabv2 import get_deeplab_v2
# from sifa_model import Encoder
# from SIFA_Model import SIFA
import argparse
from config import get_config
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Supervision_CNN_Trans2D', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument('--lr_seg', type=float,  default=0.00005,
                    help='segmentation network learning rate')
parser.add_argument('--lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)



# def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
def net_factory(net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v1":
        net = MCNet2d_v1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v2":
        net = MCNet2d_v2(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "mcnet2d_v3":
        net = MCNet2d_v3(in_chns=in_chns, class_num=class_num).cuda()
    # elif net_type == "sifamodel":
    #     net = Encoder(inputdim =1).cuda()
    # elif net_type == "sifa":
    #     net =  SIFA(args).cuda()

    elif net_type == "deeplabv2":
        RESTORE_FROM = './networks/DeepLab_resnet_pretrained_imagenet.pth'
        # RESTORE_FROM = restore
        net = get_deeplab_v2(num_classes=5, multi_level=True).cuda()
        saved_state_dict = torch.load(RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in RESTORE_FROM:
            new_params = net.state_dict().copy()

            for i in saved_state_dict:

                i_parts = i.split('.')

                if not i_parts[1] == 'layer5':

                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

            net.load_state_dict(new_params)

        else:
            net.load_state_dict(saved_state_dict)

        print('Model loaded')

    elif net_type == "deeplabv2_ours":
        RESTORE_FROM = './networks/DeepLab_resnet_pretrained_imagenet.pth'

        net = get_deeplab_v2_ours(num_classes=5).cuda()
        saved_state_dict = torch.load(RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in RESTORE_FROM:
            new_params = net.state_dict().copy()

            for i in saved_state_dict:

                i_parts = i.split('.')  # 以.符号隔开
                if not i_parts[1] == 'layer5':

                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

            net.load_state_dict(new_params)

        else:
            net.load_state_dict(saved_state_dict)

        print('Model loaded')

    return net



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    model = net_factory(net_type='deeplabv2', in_chns=3, class_num=5)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb;

    ipdb.set_trace()

