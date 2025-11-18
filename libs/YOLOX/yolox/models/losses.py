#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class CIoULoss(nn.Module):
    def __init__(self, reduction="none"):
        super(CIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Convert to top-left / bottom-right
        pred_tl = pred[:, :2] - pred[:, 2:] / 2
        pred_br = pred[:, :2] + pred[:, 2:] / 2
        target_tl = target[:, :2] - target[:, 2:] / 2
        target_br = target[:, :2] + target[:, 2:] / 2

        # Intersection
        tl = torch.max(pred_tl, target_tl)
        br = torch.min(pred_br, target_br)
        inter = (br - tl).clamp(min=0)
        area_i = inter[:, 0] * inter[:, 1]

        # Union
        area_p = pred[:, 2] * pred[:, 3]
        area_g = target[:, 2] * target[:, 3]
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-7)

        # Center distance
        center_dist = ((pred[:, :2] - target[:, :2]) ** 2).sum(dim=1)

        # Enclosing box diagonal
        c_tl = torch.min(pred_tl, target_tl)
        c_br = torch.max(pred_br, target_br)
        c_diag = ((c_br - c_tl) ** 2).sum(dim=1) + 1e-7

        # Aspect ratio consistency
        v = (4 / (3.14159265 ** 2)) * (torch.atan(target[:, 2] / target[:, 3]) - torch.atan(pred[:, 2] / pred[:, 3])) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        loss = 1 - iou + center_dist / c_diag + alpha * v

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class L1LossCostum(nn.Module):
    def __init__(self, reduction="none", normalize=False, eps=1e-6):
        super(L1LossCostum, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.normalize=normalize

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have the same shape"
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        if self.normalize:
            # x, y normalized by w+h, w,h normalized by themselves
            norm = target[:, 2:] + target[:, 3:] + self.eps  # avoid division by zero
            loss_xy = torch.abs(pred[:, :2] - target[:, :2]) / norm
            #loss_wh = torch.abs(pred[:, 2:] - target[:, 2:]) / norm
            #loss = torch.cat([loss_xy, loss_wh], dim=1)
        else:
            loss_xy = torch.abs(pred[:, :2] - target[:, :2]) 
            #loss_wh = torch.abs(pred[:, 2:] - target[:, 2:]) 
            #loss = torch.cat([loss_xy, loss_wh], dim=1)
        loss=loss_xy

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class L2LossCostum(nn.Module):
    def __init__(self, reduction="none", normalize=False, eps=1e-6):
        super(L2LossCostum, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.normalize=normalize

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have the same shape"
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        #pred_clamped = torch.clamp(pred, 0.0, 1.0)
        #pred=pred_clamped

        if self.normalize:
            # x, y normalized by w+h, w,h normalized by themselves
            norm = target[:, 2:] + target[:, 3:] + self.eps  
            loss_xy = ((pred[:, :2] - target[:, :2]) / norm) ** 2
            #loss_wh = ((pred[:, 2:] - target[:, 2:]) / norm) ** 2
            #loss = torch.cat([loss_xy, loss_wh], dim=1)
        else:
            loss_xy = (pred[:, :2] - target[:, :2]) ** 2
            #loss_wh = ((pred[:, 2:] - target[:, 2:])) ** 2
            #loss = torch.cat([loss_xy, loss_wh], dim=1)
        loss=loss_xy

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class PixelErrorLoss(nn.Module):
    """Only used for monitoring not as a loss for backpropagation"""
    
    def __init__(self, reduction="none", eps=1e-6):
        super(PixelErrorLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have the same shape"
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        cx_pred = pred[:, 0]
        cy_pred = pred[:, 1]
        cx_gt = target[:, 0]
        cy_gt = target[:, 1]

        loss = torch.sqrt((cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
