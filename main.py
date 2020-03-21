import numpy as np
import cv2
import os
import time
import torch
import torch.nn.functional as F

from utils.loss_functions import Required_Losses
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


def classification_loss(rpn_score, gt_rpn_score):
    return F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

def regression_loss(gt_rpn_score,rpn_loc,gt_rpn_loc):
    pos = gt_rpn_score > 0
    mask = pos.unsqueeze(1).expand_as(rpn_loc)
    print(mask.shape)
    # %%
    mask_loc_preds = rpn_loc[mask].view(-1, 4)
    mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
    print(mask_loc_preds.shape, mask_loc_preds.shape)
    # %%
    x = torch.abs(mask_loc_targets.float() - mask_loc_preds)
    rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))
    print('rpn loc loss',rpn_loc_loss.sum())
    N_reg = (gt_rpn_score > 0).float().sum()
    rpn_loc_loss = rpn_loc_loss.sum() / N_reg
    return rpn_loc_loss


def RPN_Loss(rpn_loc_loss,rpn_cls_loss,rpn_lambda = 10.):
    rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
    return rpn_loss


ratios = np.asarray([0.5, 1, 2])
scales = np.asarray([8, 16, 32])
bbox=np.array([[200,300,500,600]])

req_loss = Required_Losses()

gp=generate_priors(sub_sample=16,ratios=ratios,scales=scales)

gp.generate_centres()
centres1=gp.centres
anchs=gp.find_anchorV2()
pa=process_anchors(anchors= anchs,bbox= bbox,img_size=800)
print(type(pa.labels),type(pa.anchors),pa.anchors.shape)
ancs=pa.anchors
print((anchs==ancs).all())

valid_ancs= pa.find_valid_anchors(anchs) ## filter-out the out-of-img boxes!
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
print('!!!!! ground-truth',gtlocs.shape,gtlabs.shape,np.unique(gtlabs,return_counts=True))




st=time.time()

lr = 5e-4
model=BackboneRpn()
optimizer = torch.optim.SGD(model.parameters(), 
                             lr=lr,
                             weight_decay=2e-4)

model.train()
print('started training...')

for _ in range(2):

    optimizer.zero_grad()

    # img,bbox=data_loader() # load i/p img and o/p bboxes
    img, bbox = torch.randn((1, 3, 800, 800)).float() , np.array([[200,300,500,600]])
# from hre upto next para u prepare the ground_truths.!!
    ious=pa.ious
    pa.find_iou(valid_ancs,bbox)
    pa.assign_labels_and_boxes(ious)
    tab=transform_anchors_bboxes(valid_ancs,pa.box_labels)
    ancor_locs=tab.parameterize()
    mto=map_to_original_anchors(ancor_locs,pa.labels,pa.index_inside,num_anchors=22500)
    gtlocs,gtlabs=mto.map()
    gt_mini_score = mini_batch(gtlabs)
    gt_rpn_score,gt_rpn_loc = torch.from_numpy(gt_mini_score), torch.from_numpy(gtlocs)

# from here u feed-forward and then reshape the o/ps accordingly.!
    predancs,predlabs=model(img)
    rpn_locs = predancs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)[0]
    rpn_labs = predlabs.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)[0]

# from here upto next para , calculate the loss.!
    C_loss = req_loss.classification_loss(rpn_labs, gt_rpn_score)
    R_loss = req_loss.regression_loss(gt_rpn_score,rpn_locs,gt_rpn_loc)
    Loss = req_loss.RPN_Loss(R_loss,C_loss,rpn_lambda = 10.)
    Loss.backward()
    optimizer.step()

    print("Losses of RPN are ",Loss, C_loss,R_loss)
