from RPN.MobileNetV2 import MobileNetV2
from RPN.rpn_layer import rpn
import torch
import torch.nn as nn

# net = MobileNetV2(n_class=1000)
# path='C:/Users/Dell/PycharmProjects/pytorch-cnn-visualizations/src/mobilenet_v2.pth.tar'
# state_dict = torch.load(path,map_location='cpu')
# net.load_state_dict(state_dict)

class backbone_plus_rpn(nn.Module):
	def __init__(self,net,rpn,img_size=800,sub_sample=16):
		super().__init__()
		self.net=net
		self.backbone=None
		self.rpn = rpn()
		self.img_size = img_size
		self.sub_sample = sub_sample

	def create_backbone(self):
		dummy_img = torch.zeros((1, 3, 800, 800)).float()

		fe = list(self.net.features)
		req_features = []
		k = dummy_img.clone()
		for i in fe:
		    k = i(k)
		    #print(k.size())
		    if k.size()[2] < self.img_size//self.sub_sample:
		        break
		    req_features.append(i)
		    out_channels = k.size()[1]

		self.backbone = nn.Sequential(*req_features)
		# self.backbone= backbone
		# return backbone

	def forward(self,x):
		x = self.backbone(x)
		pred_ancs,pred_labs= self.rpn(x)
		return pred_ancs,pred_labs


