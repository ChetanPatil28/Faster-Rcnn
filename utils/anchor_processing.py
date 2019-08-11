import numpy as np

class process_anchors:
    def __init__(self,anchors,bbox,img_size):
        self.anchors=anchors
        self.img_size = img_size
        self.index_inside = None
        self.ious=None
        self.labels=None
        self.box_labels = None
        self.bbox=bbox

    def find_valid_anchors(self,anchors):
        index_inside = np.where((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= self.img_size) & (anchors[:, 3] <= self.img_size)) [0]
        self.index_inside = index_inside
        self.labels = np.zeros(len(index_inside))

        return anchors[index_inside] # roughly 8940 anchors

    def find_iou(self,valid_anchor_boxes,bbox): # this will 
        iou=np.zeros(shape=(len(valid_anchor_boxes),len(bbox)),dtype=np.float32)
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

    def assign_labels_and_boxes(self,ious=0): ## this will take 8940 ious
        # self.ious=ious
        pos_iou_threshold  = 0.7
        neg_iou_threshold = 0.3
        gt_argmax_ious = ious.argmax(axis=0)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(self.index_inside)), argmax_ious]
        self.labels[max_ious < neg_iou_threshold] = 0
        # Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box [a]
        self.labels[gt_argmax_ious] = 1
        # Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold [b]
        self.labels[max_ious >= pos_iou_threshold] = 1
        max_iou_bbox = self.bbox[argmax_ious]
        self.box_labels=max_iou_bbox
