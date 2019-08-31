
import torch
import torch.nn.functional as F


class Required_Losses:

    def classification_loss(self,rpn_score, gt_rpn_score):
        return F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)

    def regression_loss(self,gt_rpn_score,rpn_loc,gt_rpn_loc):   
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


    def RPN_Loss(self,rpn_loc_loss,rpn_cls_loss,rpn_lambda = 10.):
        rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
        return rpn_loss
