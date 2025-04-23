from .dataset import DOAI_Dataset
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger


def get_dataloader_from_args(phase, **kwargs):

    dataset_inst = DOAI_Dataset(
        dataset_name=kwargs['dataset'],
        category=kwargs['class_name'],
        input_size=kwargs['img_cropsize'],
        phase=phase,
        load_memory=kwargs['load_memory'],
        perturbed=kwargs['MOM'] # if MOM is applied, the data should be random perturbed
    )

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    logger.info(debug_str)

    return data_loader, dataset_inst