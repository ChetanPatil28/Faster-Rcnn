import torch
import numpy as np
import cv2
from torch.nn import Sequential
import torch.nn.functional as F
from itertools import product



class generate_priors(sub_sample,ratios,scales): ### u can give your own scalsess or be it defaul. Decide it later and based on that change the code
    def __init__(self):
        self.ratios = np.asarray([0.5, 1, 2])
        self.scales = np.asarray([8, 16, 32])
        self.sub_sample=16
        self.img_size=800
        self.fe_size=self.img_size//self.sub_sample
        # self.anchors=None
        self.centres= None
        # self.centres

    def generate_centres():
        # fe_size=img_size//sub_sample
        p=list(product(np.arange(self.sub_sample, (self.fe_size+1) * self.sub_sample, self.sub_sample),repeat=2))
        centres=np.asarray(p)-self.sub_sample/2
        self.centres = centres[:,::-1]
        # return centres[:,::-1]

    def find_anchorV2(ratios=self.ratios,scales=self.scales,sub_sample=self.sub_sample):
        self.generate_centres()
        cy=np.repeat(self.centres[:,0],9)
        cx=np.repeat(self.centres[:,1],9)
        anchors = np.zeros((len(ratios) * len(scales)*self.fe_size**2, 4), dtype=np.int16)
        scales_per_anchor=np.tile(scales,3)
        ratio_per_anchor=np.repeat(ratios,3)
        full_scale=np.tile(scales_per_anchor,2500)
        full_ratio=np.tile(ratio_per_anchor,2500)
        #print(full_ratio.shape)
        h=sub_sample*full_scale*np.sqrt(full_ratio)
        w=sub_sample*full_scale*np.sqrt(1/full_ratio)
        anchors[:,0] = cy - h / 2.
        anchors[:,1] = cx - w / 2.
        anchors[:,2] = cy + h / 2.
        anchors[:,3] = cx + w / 2.
        # self.anchors = anchors
        return anchors


class process_anchors(anchors,bbox,img_size):
    def __init__(self):
        self.anchors=anchors
        self.img_size = img_size
        self.index_inside = None
        self.ious=None
        self.labels=None
        self.box_labels = None

    def find_valid_anchors(anchors=self.anchors):
        index_inside = np.where((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= self.img_size) & (anchors[:, 3] <= self.img_size)) [0]
        self.index_inside = index_inside
        self.labels = np.zeros(len(index_inside))

        return anchors[index_inside] # roughly 8940 anchors

    def find_iou(num_anchors=0,num_bbox=0): # this will 
        iou=np.zeros(shape=(num_anchors,num_bbox),dtype=np.float32)
        for i in range(len(bbox)):
            print(bbox[i])
            min_xy = np.minimum(valid_anchor_boxes[:,2:], bbox[i,2:])
            max_xy = np.maximum(valid_anchor_boxes[:,:2], bbox[i,:2])
            inter=(max_xy[:,0]-min_xy[:,0])*(max_xy[:,1]-min_xy[:,1])
            valid_ones=np.where((max_xy[:, 0] < min_xy[:, 0]) & (max_xy[:, 1] < min_xy[:, 1]) )[0]
            # print(valid_ones.shape)
            area1=(valid_anchor_boxes[:,2]-valid_anchor_boxes[:,0])*(valid_anchor_boxes[:,3]-valid_anchor_boxes[:,1])
            area2=(bbox[i,2]-bbox[i,0])*(bbox[i,3]-bbox[i,1])
            iouu=inter/(area1+area2-inter)
            iouu=iouu[valid_ones]
            iou[valid_ones,i]=iouu
            self.ious=iou
        # return iou

    def assign_labels_and_boxes(self.ious): ## this will take 8940 ious
        pos_iou_threshold  = 0.7
        neg_ iou_threshold = 0.3
        gt_argmax_ious = ious.argmax(axis=0)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(index_inside)), argmax_ious]
        self.labels[max_ious < neg_iou_threshold] = 0
        # Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box [a]
        self.labels[gt_argmax_ious] = 1
        # Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold [b]
        self.labels[max_ious >= pos_iou_threshold] = 1
        max_iou_bbox = bbox[argmax_ious]
        self.box_labels=max_iou_bbox


class transform_anchors_bboxes(anchors=None,bboxes=None): # this will need those max_iou_bbox and valid_anchors both of size 8940 & return array of size 8940
    self.height=None
    self.width=None
    self.ctr_x=None
    self.ctr_y=None
    self.base_height=None
    self.base_width=None
    self.base_ctr_x=None
    self.base_ctr_y=None
    self.anchors = anchors
    self.bboxes= bboxes

    def convert_boxes(boxes):
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        ctr_y = boxes[:, 0] + 0.5 * base_height
        ctr_x = boxes[:, 1] + 0.5 * base_width
        return height, width,ctr_y,ctr_x

    def parameterize():
        h,w,cy,cx = self.convert_boxes(self.anchors)
        base_h,base_w,base_cy,base_cx = self.convert(self.bboxes)
        eps = np.finfo(np.float16).eps
        height = np.maximum(h, eps)
        width = np.maximum(w, eps)
        dy = (base_cy - cy) / h
        dx = (base_cx - cx) / w
        dh = np.log(base_he / h)
        dw = np.log(base_w / w)
        anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
        # print(anchor_locs.shape)
        return anchor_locs

class map_to_original_anchors(anchor_locs,labels,index_inside,num_anchors=2500):
    def __init__(self):
        self.anchor_locs=anchor_locs
        self.labels = labels
        self.index_inside = index_inside
        self.num_anchors = num_anchors

    def map():
        # Final labels:
        anchor_labels = np.empty((self.num_anchors,), dtype=self.labels.dtype)
        anchor_labels.fill(-1)
        anchor_labels[self.index_inside] = self.labels


        # Final locations

        anchor_locations = np.empty((self.num_anchors,) + 4, dtype=self.anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[self.index_inside, :] = self.anchor_locs













