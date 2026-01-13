# test CDO
import argparse
import os

import lpips_invert
import numpy as np
from tqdm import tqdm
import cv2
from datasets import *
from datasets.dataset import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from torchvision.transforms import transforms

loss_fn_vgg_inversion = lpips_invert.LPIPS(net='vgg',
                                           spatial=True,
                                           lpips=True)  # closer to "traditional" perceptual loss, when used for
loss_fn_alex_inversion = lpips_invert.LPIPS(net='alex',
                                            spatial=True,
                                            lpips=True)  # closer to "traditional" perceptual loss, when used for
loss_fn_squeeze_inversion = lpips_invert.LPIPS(net='squeeze',
                                               spatial=True,
                                               lpips=True)  # closer to "traditional" perceptual loss, when used for
# save the model, images, scores after each epoch
# save the threshold, correlation for CDO
# save the threshold, correlation after inpainintg
def test_stage_1(model, data,device):
    data = data.to(device)
    outputs = model(data)
    initial_heatmap = model.cal_am(**outputs)
    return initial_heatmap


def test_stage_2(model, data,mask,device):
    data = data.to(device)
    z = torch.from_numpy(np.random.randn(data.shape[0], model.z_dim)).to(device)
    label = torch.zeros([4, model.c_dim], device=device)
    truncation_psi=1
    noise_mode='const'
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(1)



    # outputs = (outputs.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8)
    # outputs = (outputs.permute(0, 3, 1, 2))
    # data = (data.permute(0, 3, 1, 2))

    transform_norm = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((512,512))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = transform_norm(data)
    mask = transform_norm(mask)
    outputs = model(data, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode).cpu()

    # dif = loss_fn_vgg_inversion(outputs, data)
    # im1_tensor = transform_norm(data).unsqueeze(0)
    # im2_tensor = transform_norm(outputs).unsqueeze(0)
    # dif = loss_fn_vgg_inversion(im1_tensor, im2_tensor)[1][1].detach().numpy().squeeze()
    # dif = loss_fn_alex_inversion(outputs.cpu(), data.cpu())[1][2][0].detach().cpu().numpy()
    # dif = loss_fn_squeeze_inversion(outputs.cpu(), data.cpu())[1][2][0].detach().cpu().numpy()
    dif = loss_fn_squeeze_inversion(outputs.cpu(), data.cpu())[1][2].squeeze(1).detach().cpu().numpy()
    # import cv2
    # cv2.imshow('s', np.array(cv2.cvtColor((data.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).cpu().numpy()[0], cv2.COLOR_RGB2BGR),dtype=np.uint8))
    # cv2.waitKey()

    # return outputs.cpu().numpy()
    normalized_imgs=(outputs.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).cpu().numpy()
    return dif, normalized_imgs

def test_epoch(model_ST: CDOModel,model_inpaint, dataloader: DataLoader, device: str, is_vis, img_dir):
    # change the model into eval mode
    model_ST.eval_mode()

    initial_heatmaps = None
    final_heatmaps = None
    normalized_imgs=None
    ratings=None
    test_imgs = []
    names = []
    detect=True
    if detect:
        for (data, mask, label, name) in tqdm(dataloader):
            if mask.shape[1]==4:
                e=0
            print(name)
            for d, n in zip(data, name):
                test_imgs.append((d.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
                names.append(n)


            initial_heatmap=np.array(test_stage_1(model_ST, data, device))*100
            

            mask_heatmap=True
            if mask_heatmap:
                res=initial_heatmap.shape[1]
                import cv2

                #m = cv2.imread('masks/face_mask.png', 0)
                m = cv2.imread('masks/mask.png', 0)
                m = cv2.resize(m, (res, res))
                initial_heatmap[:,m == 0] = 0

            
            T=50 #cleft
            T=25 #cleft
            initial_heatmap_b = (np.array(initial_heatmap) < T).astype(np.int_)
            # cv2.imshow('s', np.array(initial_heatmap_b[0]*255, dtype=np.uint8))
            # cv2.waitKey()
            final_heatmap,normalized_img=test_stage_2(model_inpaint,data,initial_heatmap_b, device)
            rating=[-np.sum(hmap) for hmap in final_heatmap]


            min_mean_predicted=-1500
            max_mean_predicted=0
            max_mean=4.734
            min_mean=1.448

            rating = [((rt - min_mean_predicted) / (max_mean_predicted - min_mean_predicted)) * (
                    max_mean - min_mean) + min_mean for rt in rating]
            # print(rating)
            # scores_names = [[val[1], ((-np.log10(val[2]) - min_mean_predicted) / (max_mean_predicted - min_mean_predicted)) * (
            #         max_mean - min_mean) + min_mean] for val in maps]

          
            final_heatmap=final_heatmap*255
            viz_final_map=False
            if viz_final_map:
                import cv2
                # cv2.imshow('s', cv2.cvtColor(final_heatmap[0],cv2.COLOR_RGB2BGR))
                cv2.imshow('s', np.array(final_heatmap[0],dtype=np.uint8))
                cv2.waitKey()

            if initial_heatmaps is None:
                initial_heatmaps = []

            if final_heatmaps is None:
                final_heatmaps = []


            if normalized_imgs is None:
                normalized_imgs = []

            if ratings is None:
                ratings = []

            initial_heatmaps.extend(initial_heatmap)
            final_heatmaps.extend(final_heatmap)
            normalized_imgs.extend(normalized_img)
            ratings.extend(rating)
            





    if is_vis:

        plot_sample_cv2(names, test_imgs,final_heatmaps, {'DO': initial_heatmaps}, normalized_imgs, save_folder=img_dir)




    result_dict = {'names': names, 'heatmaps': initial_heatmaps, 'ratings':ratings}

    return result_dict


def main(args):
    kwargs = vars(args)

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

    # get the model
    model = get_model_from_args(**kwargs)
    model = model.to(device)



    # directly utilize existing model for evaluation

    model_load_path = os.path.join('pretrained', f'{model_name}.pt')

    # try:
    model.load(model_load_path)

    model_inpainting = load_inpainting_model(dataset.inpainting_pkl, device)

    metrics = test_epoch(model, model_inpainting, test_dataloader, device, True, img_dir)
    print(metrics.get("ratings"))

    return metrics

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
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--heatmaps_stage1", type=str, default="./heatmaps_stage1")
    parser.add_argument("--heatmaps_stage2", type=str, default="./heatmaps_stage2")
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
