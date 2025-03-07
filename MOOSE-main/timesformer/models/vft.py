import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

class VisualFlowTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def get_spatial_embed(self, x):
        return x
    
    def get_optical_flow(self, x):
        return x
    
    def forward(self, x):
        return x