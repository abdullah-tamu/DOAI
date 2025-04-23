import random
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

from models import *
from utils.visualization import *
import dnnlib
from networks.mat import Generator
import legacy
import copy

def normalize(v):
    return (v - np.min(v)) / (np.max(v) - np.min(v))
def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

def load_inpainting_model(inpainting_pkl,device):
    resolution = 512
    print(f'Loading networks from: {inpainting_pkl}')
    # device = torch.device('cuda')
    with dnnlib.util.open_url(inpainting_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
    net_res = 512 if resolution > 512 else resolution
    model_inpainting = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(
        device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, model_inpainting, require_all=True)
    model_inpainting = copy.deepcopy(model_inpainting).eval().requires_grad_(False).to(device).float()  # type: ignore
    return model_inpainting

def get_tensorboard_logger_from_args(tensorboard_dir, reset_version=False):
    if reset_version:
        shutil.rmtree(os.path.join(tensorboard_dir))
    return SummaryWriter(log_dir=tensorboard_dir)


def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(root_dir, class_name, backbone, **kwargs):
    exp_name = f"{backbone}-{kwargs['dataset']}"

    if kwargs['OOM']:
        exp_name = f"{exp_name}-wi-OOM"
    else:
        exp_name = f"{exp_name}-wo-OOM"

    if kwargs['MOM']:
        exp_name = f"{exp_name}-wi-MOM"
    else:
        exp_name = f"{exp_name}-wo-MOM"
    # exp_name=f'{exp_name}_256'
    csv_dir = os.path.join(root_dir, 'csv')
    csv_path = os.path.join(csv_dir, f"{exp_name}.csv")

    # model_dir = os.path.join(root_dir, exp_name, 'models_g_2')
    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'initial_heatmaps')

    tensorboard_dir = os.path.join(root_dir, 'tensorboard', exp_name, class_name)
    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    log_file_name = os.path.join(logger_dir,
                                 f'log_{time.strftime("%Y-%m-%d-%H-%I-%S", time.localtime(time.time()))}.log')

    model_name = f'{class_name}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    logger.start(log_file_name)

    logger.info(f"===> Root dir for this experiment: {logger_dir}")

    return model_dir, img_dir, tensorboard_dir, logger_dir, model_name, csv_path
