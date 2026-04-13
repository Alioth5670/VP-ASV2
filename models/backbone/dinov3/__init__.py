
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from .vision_transformer import DinoVisionTransformer


class DINOv3Backbone(nn.Module):
    '''
    name: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, dinov3_vitl16
    weight_path: path to the weight file
    interaction_indexes: list of interaction indexes
    '''
    def __init__(self, 
                 name, 
                 weight_path=None, 
                 interaction_indexes=[],
                 freeze=True):
        super().__init__()
        print(f"[DINOv3Backbone] init start: name={name}", flush=True)
        if 'dinov3' in name:
            self.vision_transformer = DinoVisionTransformer(name)
        else:
            raise ValueError(f"Unknown model name: {name}")
        print(f"[DINOv3Backbone] vision_transformer created: name={name}", flush=True)
        if weight_path is not None:
            print(f"Loading weight from {weight_path}", flush=True)
            t0 = time.time()
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.vision_transformer.load_state_dict(state_dict)
            print(f"Loaded weight from {weight_path} in {time.time() - t0:.2f}s", flush=True)
            if freeze:
                self.vision_transformer.eval()
                for param in self.vision_transformer.parameters():
                    param.requires_grad = False
        else:
            if freeze:
                raise ValueError("freeze must be False when training from scratch")
            print(f"Training {name} from scratch")
        self.patch_size = self.vision_transformer.patch_size
        self.interaction_indexes = interaction_indexes
    
    def forward(self, x):
        if len(self.interaction_indexes) > 0:
            features = self.vision_transformer.get_intermediate_layers(x, 
                                                                       n=self.interaction_indexes,
                                                                       return_class_token=True)
        else:
            f = self.vision_transformer(x)
            features = [f]
        return features
        
