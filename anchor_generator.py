import torch
import numpy as np
import cv2
from torch.nn import Sequential
import torch.nn.functional as F
from itertools import product


ratios = np.asarray([0.5, 1, 2])
scales = np.asarray([8, 16, 32])


def generate_centres(img_size=800,sub_sample=16):
    fe_size=img_size//sub_sample
    p=list(product(np.arange(16, (fe_size+1) * sub_sample, sub_sample),repeat=2))
    centres=np.asarray(p)-sub_sample/2
    return centres[:,::-1]

def find_anchor(cx,cy,ratios,scales,sub_sample=16):
    anchor_basee = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)
    scales_per_anchor=np.tile(scales,3)
    ratio_per_anchor=np.repeat(ratios,3)
    h=sub_sample*scales_per_anchor*np.sqrt(ratio_per_anchor)
    w=sub_sample*scales_per_anchor*np.sqrt(1/ratio_per_anchor)
    anchor_basee[:,0]=cy - h / 2.
    anchor_basee[:,1]=cx - w / 2.
    anchor_basee[:,2]=cy + h / 2.
    anchor_basee[:,3]=cx + w / 2.
    return anchor_basee

def find_anchorV2(centres,ratios,scales,sub_sample=16):
    cy=np.repeat(centres[:,0],9)
#     print(cx.shape)
    cx=np.repeat(centres[:,1],9)
    anchor_basee = np.zeros((len(ratios) * len(scales)*2500, 4), dtype=np.int16)
    scales_per_anchor=np.tile(scales,3)
    ratio_per_anchor=np.repeat(ratios,3)
    full_scale=np.tile(scales_per_anchor,2500)
    full_ratio=np.tile(ratio_per_anchor,2500)
    #print(full_ratio.shape)
    h=sub_sample*full_scale*np.sqrt(full_ratio)
    w=sub_sample*full_scale*np.sqrt(1/full_ratio)
    anchor_basee[:,0]=cy - h / 2.
    anchor_basee[:,1]=cx - w / 2.
    anchor_basee[:,2]=cy + h / 2.
    anchor_basee[:,3]=cx + w / 2.
    return anchor_basee
