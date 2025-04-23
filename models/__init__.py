from models.ST import *
from loguru import logger

def get_model_from_args(**kwargs)->CDOModel:
    model = CDOModel(**kwargs)
    return model