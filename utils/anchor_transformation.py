import numpy as np

class transform_anchors_bboxes: # this will need those max_iou_bbox and valid_anchors both of size 8940 & return array of size 8940
    def __init__(self,anchors=None,bboxes=None):
        self.anchors = anchors
        self.bboxes= bboxes

    def convert_boxes(self,boxes):
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        ctr_y = boxes[:, 0] + 0.5 * height
        ctr_x = boxes[:, 1] + 0.5 * width
        return height, width,ctr_y,ctr_x

    def parameterize(self):
        h,w,cy,cx = self.convert_boxes(self.anchors)
        base_h,base_w,base_cy,base_cx = self.convert_boxes(self.bboxes)
        eps = np.finfo(np.float16).eps
        height = np.maximum(h, eps)
        width = np.maximum(w, eps)
        dy = (base_cy - cy) / h
        dx = (base_cx - cx) / w
        dh = np.log(base_h / h)
        dw = np.log(base_w / w)
        anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
        # print(anchor_locs.shape)
        return anchor_locs

class map_to_original_anchors:
    def __init__(self,anchor_locs,labels,index_inside,num_anchors=22500):
        self.anchor_locs=anchor_locs
        self.labels = labels
        self.index_inside = index_inside
        self.num_anchors = num_anchors

    def map(self):
        # Final labels:
        anchor_labels = np.empty((self.num_anchors,), dtype=self.labels.dtype)
        anchor_labels.fill(-1)
        anchor_labels[self.index_inside] = self.labels


        # Final locations

        anchor_locations = np.empty((self.num_anchors,4) , dtype=self.anchor_locs.dtype)
        anchor_locations.fill(0)
        anchor_locations[self.index_inside, :] = self.anchor_locs
        return anchor_locations,anchor_labels

        # Final locations

        # anchor_locations = np.empty((self.num_anchors,) + 4, dtype=self.anchor_locs.dtype)
        # anchor_locations.fill(0)
        # anchor_locations[self.index_inside, :] = self.anchor_locs