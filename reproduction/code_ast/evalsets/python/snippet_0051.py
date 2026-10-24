from fastai.vision import ImageDataBunch,get_transforms,imagenet_stats
import numpy as np
import pdb
def create_data_bunch(path):
    np.random.seed(42)
    return ImageDataBunch.from_folder(path,train=".",valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)