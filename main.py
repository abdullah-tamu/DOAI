from test_DOAI import test_epoch
# test stage 1

# inpaint MAT


# test CDO
import argparse
import os

from tqdm import tqdm
from datasets import *
from datasets.dataset import dataset_classes
from utils.csv_utils import *
# from utils.metrics import *
from utils.training_utils import *
import dnnlib
from networks.mat import Generator
import legacy
import copy


# save the model, images, scores after each epoch
# save the threshold, correlation for CDO
# save the threshold, correlation after inpainintg


def main(args):
    kwargs = vars(args)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    setup_seed(kwargs['seed'])
    device = f"cuda:{kwargs['gpu_id']}"
    kwargs['device'] = device

    # prepare the experiment dir
    model_dir, img_dir, tensorboard_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    h, w = test_dataset_inst.get_size()
    kwargs['out_size_h'] = h
    kwargs['out_size_w'] = w

    # get the model_ST
    model_ST = get_model_from_args(**kwargs)
    model_ST = model_ST.to(device)

    customize = False
    if customize:
        epoch = 25
        res = kwargs['out_size_h']
        model_dir_2 = model_dir.split('/')
        model_dir_2[-2] = model_dir_2[-2] + f'_{str(res)}'
        model_dir_2 = '/'.join(model_dir_2)
        model_name_2 = model_name + f'_{epoch}'

        img_dir_2 = img_dir.split('/')
        img_dir_2[-2] = img_dir_2[-2] + f'_{str(res)}'
        img_dir_2 = '/'.join(img_dir_2)

        model_dir = model_dir_2
        model_name = model_name_2
        img_dir = img_dir_2
        img_dir = './initial_heatmaps/'
        # os.removedirs(img_dir)
        shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)

    # directly utilize existing model_ST for evaluation
    model_load_path = os.path.join(model_dir, f'{model_name}.pt')
    model_load_path = os.path.join('pretrained', f'{model_name}.pt')

    # try:
    model_ST.load(model_load_path)

    inpainting_pkl='pretrained/FFHQ_512.pkl'
    # inpainting_pkl='pretrained/CelebA-HQ_512.pkl'
    model_inpainting=load_inpainting_model(inpainting_pkl,device)

    metrics = test_epoch(model_ST,model_inpainting, test_dataloader, device, True, img_dir)



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec2d', choices=['mvtec2d', 'mvtec3d'])
    parser.add_argument('--class-name', type=str, default='face')
    parser.add_argument('--img-resize', type=int, default=512)
    parser.add_argument('--img-cropsize', type=int, default=512)

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument("--validation-epoch", type=int, default=5)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument("--backbone", type=str, default="hrnet48",
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'hrnet18', 'hrnet32',
                                 'hrnet48'])
    parser.add_argument("--MOM", type=str2bool, default=True)
    parser.add_argument("--OOM", type=str2bool, default=True)
    parser.add_argument("--gamma", type=float, default=2.)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
