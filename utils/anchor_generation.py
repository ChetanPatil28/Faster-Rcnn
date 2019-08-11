from itertools import product
import numpy as np


ratios = np.asarray([0.5, 1, 2])
scales = np.asarray([8, 16, 32])


class generate_priors: ### u can give your own scalsess or be it defaul. Decide it later and based on that change the code
    def __init__(self,sub_sample,ratios,scales,img_size=800):
        self.ratios = ratios
        self.scales = scales
        self.sub_sample=sub_sample
        self.img_size=img_size
        self.fe_size=self.img_size//self.sub_sample
        self.centres= None

    def generate_centres(self):
        p=list(product(np.arange(self.sub_sample, (self.fe_size+1) * self.sub_sample, self.sub_sample),repeat=2))
        centres=np.asarray(p)-self.sub_sample/2
        self.centres = centres[:,::-1]

    def find_anchorV2(self): #self,ratios=self.ratios,scales=self.scales,sub_sample=self.sub_sample
        self.generate_centres()
        cy=np.repeat(self.centres[:,0],9)
        cx=np.repeat(self.centres[:,1],9)
        anchors = np.zeros((len(self.ratios) * len(self.scales)*self.fe_size**2, 4), dtype=np.int16)
        scales_per_anchor=np.tile(self.scales,3)
        ratio_per_anchor=np.repeat(self.ratios,3)
        full_scale=np.tile(scales_per_anchor,2500)
        full_ratio=np.tile(ratio_per_anchor,2500)
        #print(full_ratio.shape)
        h=self.sub_sample*full_scale*np.sqrt(full_ratio)
        w=self.sub_sample*full_scale*np.sqrt(1/full_ratio)
        anchors[:,0] = cy - h / 2.
        anchors[:,1] = cx - w / 2.
        anchors[:,2] = cy + h / 2.
        anchors[:,3] = cx + w / 2.
        return anchors


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
