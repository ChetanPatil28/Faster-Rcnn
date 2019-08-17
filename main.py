import numpy as np
import cv2
import os
import time
import torch

from utils.anchor_generation import generate_priors
from utils.anchor_processing import process_anchors
from utils.anchor_transformation import transform_anchors_bboxes,map_to_original_anchors
from RPN.propose_objects import BackboneRpn

def mini_batch(label,pos_ratio = 0.5,n_sample = 256):
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


ratios = np.asarray([0.5, 1, 2])
scales = np.asarray([8, 16, 32])
bbox=np.array([[200,300,500,600]])
gp=generate_priors(sub_sample=16,ratios=ratios,scales=scales)

gp.generate_centres()
centres1=gp.centres
anchs=gp.find_anchorV2()
pa=process_anchors(anchors= anchs,bbox= bbox,img_size=800)
print(type(pa.labels),type(pa.anchors),pa.anchors.shape)
ancs=pa.anchors
print((anchs==ancs).all())

valid_ancs= pa.find_valid_anchors(anchs)
print(type(pa.labels),pa.labels.shape,valid_ancs.shape)
print('labels uniq',np.unique(pa.labels))

pa.find_iou(valid_ancs,bbox)
ious=pa.ious
print(ious.shape,np.unique(ious))
print('XXXX',pa.bbox)
print('xx',type(pa.box_labels))
pa.assign_labels_and_boxes(ious)
print('xx',type(pa.box_labels),pa.box_labels.shape)
print(np.unique(pa.labels,return_counts=True),np.mean(pa.box_labels,axis=0))

tab=transform_anchors_bboxes(valid_ancs,pa.box_labels)
ancor_locs=tab.parameterize()
print('ancor_locs',ancor_locs.shape)
print(ancor_locs)

mto=map_to_original_anchors(ancor_locs,pa.labels,pa.index_inside,num_anchors=22500)
gtlocs,gtlabs=mto.map()
print('ground-truth',gtlocs.shape,gtlabs.shape,np.unique(gtlabs,return_counts=True))


print('start')
st=time.time()
model=BackboneRpn()
predancs,predlabs=model(torch.zeros((1, 3, 800, 800)).float())
print(predancs.shape,predlabs.shape)

print('dones',time.time()-st)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

rpn_locs = predancs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)[0]
rpn_labs = predlabs.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)[0]

gt_rpn_loc = torch.from_numpy(gtlocs)
gt_rpn_score = torch.from_numpy(gtlabs)

gt_mini_score = mini_batch(gtlabs)
print(np.unique(gt_mini_score,return_counts=True))