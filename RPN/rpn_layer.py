import torch
import torch.nn as nn

class rpn(nn.Module):
  
    def __init__ (self,in_channels = 96,mid_channels = 512):
        super().__init__()

         # depends on the output feature map. in vgg 16 it is equal to 512
        self.n_anchor = 9 # Number of anchors at each location
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, self.n_anchor *4, 1, 1, 0)
        self.cls_layer = nn.Conv2d(mid_channels, self.n_anchor *2, 1, 1, 0) 

        # conv sliding layer
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()
        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()
  
    def forward(self, x):
        
        main = self.conv1(x)
        pred_anchor_locs = self.reg_layer(main)
        pred_anchor_labs = self.cls_layer(main)

        return pred_anchor_locs,pred_anchor_labs


class reformat_predictions:

    def __init__(self,pred_anchor_locs,pred_anchor_labs):
        pass
    def convert(self,pred_anchor_locs,pred_anchor_labs):
        pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)






       