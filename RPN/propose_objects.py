from RPN.MobileNetV2 import MobileNetV2
from RPN.rpn_layer import rpn
import torch
import torch.nn as nn
imort os

mbnet = MobileNetV2(n_class=1000)
bbone_path = "mobilenet_v2.pth.tar"

path = os.path.join(os.getcwd(), bbone_path)
state_dict = torch.load(path,map_location='cpu')
mbnet.load_state_dict(state_dict)

def create_backbone(net,img_size=800,sub_sample=16):
    
    dummy_img = torch.zeros((1, 3, 800, 800)).float()
    fe = list(net.features)
    req_features = []
    k = dummy_img.clone()
    for i in fe:
        k = i(k)
        #print(k.size())
        if k.size()[2] < img_size//sub_sample:
            break
        req_features.append(i)
        out_channels = k.size()[1]

    backbone = nn.Sequential(*req_features)
    return backbone

backbone = create_backbone(mbnet)
reg_prop = rpn() 

class BackboneRpn(nn.Module):
    def __init__(self,rpnn = rpn(),net=backbone,img_size=800,sub_sample=16): ### u can put both reg_prop or rpn() here
        super().__init__()
        #self.net=net
        self.backbone= net
        self.rpn = rpnn
        # print(self.rpn())
        self.img_size = img_size
        self.sub_sample = sub_sample
 
    def forward(self,x):
        x = self.backbone(x)
        pred_ancs,pred_labs= self.rpn(x)
        return pred_ancs,pred_labs


# class backbone_plus_rpnV2(nn.Module):
#     def __init__(self,net=None,img_size=800,sub_sample=16,in_channels=96,mid_channels=512):
#         super().__init__()
#         self.net=net
#         self.backbone=None
#         #self.rpn = rpn()
#         #print(self.rpn)
#     # def create_backbone(self):
#         dummy_img = torch.zeros((1, 3, 800, 800)).float()

#         fe = list(self.net.features)
#         req_features = []
#         k = dummy_img.clone()
#         for i in fe:
#             k = i(k)
#             #print(k.size())
#             if k.size()[2] < img_size//sub_sample:
#                 break
#             req_features.append(i)
#             out_channels = k.size()[1]

#         self.backbone = nn.Sequential(*req_features)
#         #print('AAAAAAAAAA',self.backbone)

#         self.img_size = img_size
#         self.sub_sample = sub_sample
#         self.n_anchor = 9 # Number of anchors at each location
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
#         self.reg_layer = nn.Conv2d(mid_channels, self.n_anchor *4, 1, 1, 0)
#         self.cls_layer = nn.Conv2d(mid_channels, self.n_anchor *2, 1, 1, 0) 

#         # conv sliding layer
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv1.bias.data.zero_()
#         # Regression layer
#         self.reg_layer.weight.data.normal_(0, 0.01)
#         self.reg_layer.bias.data.zero_()
#         # classification layer
#         self.cls_layer.weight.data.normal_(0, 0.01)
#         self.cls_layer.bias.data.zero_()

#         # self.backbone= backbone
#         # return backbone
        

#     def forward(self,x):
#         #print('BABBA',self.backbone)

#         #self.create_backbone()
#         #print('BABABA',self.backbone)

#         x = self.backbone(x)
#         print(x.shape)
#         main = self.conv1(x)
#         print(main.shape)

#         pred_anchor_locs = self.reg_layer(main)
#         pred_anchor_labs = self.cls_layer(main)

#         return pred_anchor_locs,pred_anchor_labs

#         # pred_ancs,pred_labs= self.rpn(x)
#         # return pred_ancs,pred_labs


