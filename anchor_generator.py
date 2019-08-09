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

def find_iou(num_anchors=0,num_bbox=0):
    iou=np.zeros(shape=(num_anchors,num_bbox),dtype=np.float32)
    for i in range(len(bbox)):
        print(bbox[i])
        min_xy = np.minimum(valid_anchor_boxes[:,2:], bbox[i,2:])
        max_xy = np.maximum(valid_anchor_boxes[:,:2], bbox[i,:2])
        inter=(max_xy[:,0]-min_xy[:,0])*(max_xy[:,1]-min_xy[:,1])
        valid_ones=np.where((max_xy[:, 0] < min_xy[:, 0]) & (max_xy[:, 1] < min_xy[:, 1]) )[0]
        print(valid_ones.shape)
        area1=(valid_anchor_boxes[:,2]-valid_anchor_boxes[:,0])*(valid_anchor_boxes[:,3]-valid_anchor_boxes[:,1])
        area2=(bbox[i,2]-bbox[i,0])*(bbox[i,3]-bbox[i,1])
        iouu=inter/(area1+area2-inter)
        iouu=iouu[valid_ones]
        iou[valid_ones,i]=iouu
    return iou

def mini_batch(pos_ratio = 0.5,n_sample = 256,label):
    n_pos = pos_ratio * n_sample
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - int(n_pos)), replace=False)
        label[disable_index] = -1
    ### 2. negative_labels
    n_neg = n_sample * np.sum(label == 1)
    #n_neg=128
    ### why havent we used n_neg also as 128 ??? instead n_neg is a bigggggg number 
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
       # print("aws")
        disable_indexi = np.random.choice(neg_index, size=(len(neg_index) - int(n_neg)), replace = False)
        label[disable_indexi] = -1
    return label

def convert_boxes(boxes):
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    ctr_y = boxes[:, 0] + 0.5 * base_height
    ctr_x = boxes[:, 1] + 0.5 * base_width
    return height, width,ctr_y,ctr_x