import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(preds, targets, weight=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w x d)
        gt_regr (B x c x h x w x d)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, weight)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,4,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,1,3)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss

def nms(heat, kernel=3):
    hmax = F.max_pool3d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence, cuda):

    pred_hms = nms(pred_hms)
    b, c, output_h, output_w, output_d = pred_hms.shape
    detects = []

    for batch in range(b):

        heat_map    = pred_hms[batch].permute(1, 2, 3, 0).view([-1, c])
        pred_wh     = pred_whs[batch].permute(1, 2, 3, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 3, 0).view([-1, 3])

        zv, yv, xv  = torch.meshgrid(torch.arange(0, output_d), torch.arange(0, output_h), torch.arange(0, output_w))
        xv, yv, zv= xv.flatten().float(), yv.flatten().float(), zv.flatten().float()
        if cuda:
            xv      = xv.cuda()
            yv      = yv.cuda()

        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue     
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        zv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 2], -1)
        
        half_w, half_h, half_d = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2, pred_wh_mask[..., 2:3] / 2

        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, zv_mask - half_d, xv_mask + half_w, yv_mask + half_h, zv_mask + half_d], dim=1)
        bboxes[:, [0, 3]] /= output_w
        bboxes[:, [1, 4]] /= output_h
        bboxes[:, [2, 5]] /= output_d
        
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects
