import os
import sys

from ..datasets import VisualGenomeTrainData
from ..datasets import VRDTrainData
from detectron2.data.datasets import register_coco_instances

def register_datasets(cfg):
    if cfg.DATASETS.TYPE == 'VISUAL GENOME':
        for split in ['train', 'val', 'test']:
            dataset_instance = VisualGenomeTrainData(cfg, split=split)
    elif cfg.DATASETS.TYPE == 'VRD':
        for split in ['train', 'test']:
            dataset_instance = VRDTrainData(cfg, split=split)
        