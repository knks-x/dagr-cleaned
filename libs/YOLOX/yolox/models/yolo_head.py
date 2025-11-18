#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, visualize_assign

from .losses import IOUloss, L1LossCostum, CIoULoss, L2LossCostum, PixelErrorLoss
from .network_blocks import BaseConv, DWConv

from dagr.utils.args import FLAGS
from dagr.data.ball_utils import visualize_events_with_boxes, visualize_events_after, visualize_boxes, select_top_bboxes, save_detections_csv

args = FLAGS()

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes=1,            
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):

            #bridge between backbone and detection head: feature map into format that detection head wants (out channel)
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )

            #regression branch
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            #regression branch
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            #objectness branch 
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
                                   
        self.use_l1 = True                                                   #needs to be true when using L2 too 
        #self.l1_loss = nn.L1Loss(reduction="none")                          #original l1: computing loss of xywh of box                        
        self.l1_loss = L1LossCostum(reduction="none")
        self.l2_loss = L2LossCostum(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")        #used for cls and conf loss (objectness)
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.use_pe = True  
        self.pe_loss = PixelErrorLoss(reduction="none")                        #for monitoring the pixel error during learning (L1 relative to grid loss), no backprop

    def initialize_biases(self, prior_prob):
        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_conv, stride_this_level, x) in enumerate(                       #k not related to anchor boxes but is index of current detection SCALE =[8, 16, 32]
                                                                                    #x: feature map from backbone
            zip(self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)                                                    #pass feature map through stem conv (make ready for detection head)
            reg_x = x

            reg_feat = reg_conv(reg_x)                                              #convolutions on regression
            reg_output = self.reg_preds[k](reg_feat)                                #output layer: compute regression and objectness
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output], 1)                     #flattening the tensor along column: output=[N, 5+C, H, W]  
                                                                                        #-> every image has H*W grid cell prediction with each 5+C values (bbox_reg, obj, num_cls)
                output, grid = self.get_output_and_grid(                            #removed classes inside 
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                    #print("Debug l1, origin preds:", origin_preds)

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid()], 1
                )

            outputs.append(output)
            
        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                events=xin.pos
            )
        else:                                                                          #returns predictions
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):                           #k=[8,16,32], stride=stride at scale k
        
        """returns permutated and shaped output like this [N, H*W, (5+C)]
        to interpret output as each row = one grid cell with the follwing values 1-4-> bbox reg 5->objectness """

        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5                                                                       #no more + classes here
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:                                       #grid of x,y for each cell -> e.g. grid is shape [1, 1, 80, 80, 2]        
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)          
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)                        #1 prediction per cell, but they keep this dimension for consistency with older YOLO code. (yolo: k anchor boxes)
        output = output.permute(0, 1, 3, 4, 2).reshape(                                #position 2 moves to the end = 5+C is at the end
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,                         #necessary for bbox reg too
        outputs,
        origin_preds,
        dtype,
        events
    ):
        bbox_preds = outputs[:, :, :4]                                  #[batch, n_anchors_all, 4] most likely in cx cy format
        obj_preds = outputs[:, :, 4:5]                                  #[batch, n_anchors_all, 1]
        
        #mask out dead conf values =0.5
        
        #print(mask_obj)
        #obj_preds_filtered = obj_preds[mask_obj]
        #print(f"Length obj pred before filtering:{len(obj_preds)} and after: {len(obj_preds_filtered)}")

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)                     #shape: [label_class, cx, cy, w ,h ]
        total_num_anchors = outputs.shape[1]

        x_shifts = torch.cat(x_shifts, 1)                               #[1, n_anchors_all]. grid cell centers in the feature map for each anchor
        y_shifts = torch.cat(y_shifts, 1)                               #x and y give absolute cooordinates of the center of each grid cell/anchor
        
        expanded_strides = torch.cat(expanded_strides, 1)               #shifts are scaled by expanded_strides to map back into “pixel-like” coordinates in the feature map space
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        decoded_boxes_list = []
        strides_unique = expanded_strides.unique()

        num_fg = 0.0                                                    #num of pos anchors
        num_gts = 0.0
        img_counter=0
        save_path_e =f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/events/{img_counter}"
        save_path_t = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/top-scores/{img_counter}"
        save_path_new = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/new/pred+gt/{img_counter}"
        
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])                             #get number of gt objects for that batch
            num_gts += num_gt
            if num_gt == 0:
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]   #save that one gt in the gt_bboxes per image
                bboxes_preds_per_image = bbox_preds[batch_idx]          #save all bboxes in the bboxes per image
                obj_pred_per_image = obj_preds[batch_idx]
                
                
                #DEBUG CONFIDENCE vs LOCATION
                """if batch_idx == 5: 
                    #img_counter=0                                                                                       #batch_idx refers to sample/time window: in each batch 32 samples - 100 batches per epoch in training                                             
                    obj_pred_prob = torch.sigmoid(obj_pred_per_image)                                                    #obj has raw logits
                    obj_pred_col = obj_pred_prob.view(-1, 1)  
                    preds_per_image = torch.cat([bboxes_preds_per_image, obj_pred_col], dim=1) 
                    save_detections_csv(gt_bboxes_per_image, preds_per_image, f"outputs/training/debug_confidence-sample{batch_idx}-{img_counter}.png")
                    img_counter += 1"""

                #save top boxes for debugging
                num_boxes = obj_preds.shape[1]
                k = max(1, int(num_boxes * 0.25))
                topk_scores, topk_idx = torch.topk(obj_preds[batch_idx].squeeze(-1), k, dim=0)
                top_bboxes_per_image = bbox_preds[batch_idx][topk_idx]

                #img_counter=0
                if batch_idx % 30 == 0:
                    save_path = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/pred+gt/{img_counter}"
                    print("Visualize pred+gt before assignment")
                    visualize_events_with_boxes(events=events,
                        gt_boxes=gt_bboxes_per_image,
                        idx=batch_idx,
                        pred_boxes=bboxes_preds_per_image,
                        img_width=1280/2, img_height=720/2,
                        title=f"GT vs Predictions",
                        line_pred=0.7,
                        save_path=save_path)
                    #visualize_events_after(events=events, idx=batch_idx, save_path=save_path_e)
                    #visualize_boxes(gt_boxes=gt_bboxes_per_image, idx=batch_idx, pred_boxes=bboxes_preds_per_image, top_boxes=None, img_width=640/2, img_height=430/2, title=f"GT vs Predictions", line_pred=1.7, save_path=save_path)
                    img_counter += 1

                try:
                    (
                        fg_mask,                                        
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa                   #get_assignments returns num of and mask with pos anchors, pred ious per anchor, matching gt indices 
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        bboxes_preds_per_image,
                        topk_idx,
                        top_bboxes_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        obj_preds,
                        events,
                        img_counter
                    )
                except RuntimeError as e:                               

                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa                   
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        bboxes_preds_per_image,
                        topk_idx,
                        top_bboxes_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        obj_preds,
                        events,
                        "cpu",
                        img_counter
                    )

                pos_pred_boxes = bboxes_preds_per_image[fg_mask]
                pos_gt_boxes = gt_bboxes_per_image[matched_gt_inds]

                topk_is_fg = fg_mask[topk_idx]
                pos_top_bboxes_per_image= top_bboxes_per_image[topk_is_fg]
                if top_bboxes_per_image.numel() == 0:
                    print(f"[WARN] No top boxes left after applying full fg mask for sample {batch_idx}")

                #img_counter=0
                if batch_idx % 30 == 0:
                    print("Visualize anchor+gt after assignment (including cost, simota, iou...)")
                    save_path2 = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/pred+gt-after-mask/{img_counter}"
                    save_path2_new = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/new/pred+gt-after-mask/{img_counter}"
                    visualize_events_with_boxes(events=events, 
                                                gt_boxes=pos_gt_boxes, 
                                                pred_boxes=pos_pred_boxes, 
                                                idx=batch_idx, 
                                                #img_width=640/2,
                                                #img_height=430/2,
                                                img_width=1280/2, img_height=720/2,
                                                title=f"GT and Anchors after assignment", 
                                                line_pred=1.7,
                                                save_path=save_path2)
                    #visualize_boxes(gt_boxes=gt_bboxes_per_image, idx=batch_idx, pred_boxes=pos_pred_boxes, top_boxes=None, img_width=640/2, img_height=430/2, title=f"GT vs Predictions After Assignment", line_pred=3, save_path=save_path2)
                    img_counter += 1

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                #each anchor gets a corresponding target to get compared two
                obj_target = fg_mask.unsqueeze(-1)                      #objectness 1 at last
                """print(f"Shape: {obj_target.shape}")
                print(f"Device: {obj_target.device}, Dtype: {obj_target.dtype}")
                print(f"Min: {obj_target.min().item()}, Max: {obj_target.max().item()}")
                unique_vals = torch.unique(obj_target)
                print(f"Unique values: {unique_vals}")
                
                # Show a small slice for manual inspection
                flat = obj_target.flatten()
                n = min(len(flat), 10)
                print(f"First {n} values: {flat[:n]}")
                print("------------------\n")"""
                reg_target = gt_bboxes_per_image[matched_gt_inds]       #target gt box 
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                
            #append all targets
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        #concatenate across batch
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        """print(f"Shape: {obj_targets.shape}")
        print(f"Device: {obj_targets.device}, Dtype: {obj_targets.dtype}")
        print(f"Min: {obj_targets.min().item()}, Max: {obj_targets.max().item()}")
        unique_vals = torch.unique(obj_targets)
        print(f"Unique values: {unique_vals}")
        
        # Show a small slice for manual inspection
        flat = obj_targets.flatten()
        n = min(len(flat), 10)
        print(f"First {n} values: {flat[:n]}")
        print("------------------\n")"""

        #filter obj pred for loss computation
        mask_obj = obj_preds.view(-1, 1).abs() > 0.0
        obj_preds_filtered = obj_preds.view(-1, 1)[mask_obj]       # only live logits
        obj_targets_filtered = obj_targets[mask_obj]   

        #compute losses
        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        
        loss_obj = (
            self.bcewithlog_loss(obj_preds_filtered #obj_preds.view(-1, 1)
            , obj_targets_filtered)
            
        ).sum() / num_fg
        """
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1)
            , obj_targets)
            
        ).sum() / num_fg """
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg                                                            #sum over num of positive anchors = mean
            loss_l2 = (
                self.l2_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0
            loss_l2 = 0.0
        if self.use_pe:
            loss_pe=(
                self.pe_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                ).sum() / num_fg 
        else:
             loss_pe = 0.0


        iou_w = 5.0
        obj_w= 1.0
        l1_w = 0.0
        l2_w = 0.0
        
        loss = iou_w * loss_iou + obj_w * loss_obj +  l1_w * loss_l1  + l2_w * loss_l2        
        
        return (
            loss,
            loss_iou,
            loss_obj,
            loss_l1,
            loss_l2,
            loss_pe,
            num_fg / max(num_gts, 1),
        )
    
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        bboxes_preds_per_image,
        topk_idx,
        top_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        obj_preds,
        events,
        img_counter,
        mode="gpu"
        
    ):
        """returns: mask for all candidate anchors, IoU for positive anchors, matched gt and pos anchor, number of positive anchors"""

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(                              #fg mask: which anchors are inside or near GT boxes - anchors and gt in same pixel space
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        #debug
        #print("Before geometry filter:", len(bboxes_preds_per_image))

        topk_is_fg = fg_mask[topk_idx]
        top_bboxes_per_image= top_bboxes_per_image[topk_is_fg]
        if top_bboxes_per_image.numel() == 0:
            print(f"[WARN] No top boxes left after applying geometry filter for sample {batch_idx}")
        
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]                                #filter out not in-box anchors
        #print("After geometry filter:", len(bboxes_preds_per_image))   

        img_counter=img_counter
        save_path3 = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/pred+gt-after-filter/{img_counter}"  
        save_path3_new = f"/data/sbaumann/nature-gnn/dagr/visualize-to-debug/ball/new/pred+gt-after-filter/{img_counter}"      
        if batch_idx % 30 == 0:
            #print("Visualize anchor+gt after geometry relation")
            visualize_events_with_boxes(events=events, 
                                        gt_boxes=gt_bboxes_per_image, 
                                        pred_boxes=bboxes_preds_per_image, 
                                        idx=batch_idx, 
                                        #img_width=640/2,
                                        #img_height=430/2,
                                        img_width=1280/2, img_height=720/2,
                                        title=f"Target + Anchors after geometry pre-filter", 
                                        line_pred=0.7,
                                        save_path=save_path3)
            #visualize_boxes(gt_boxes=gt_bboxes_per_image, idx=batch_idx, pred_boxes=bboxes_preds_per_image, top_boxes=None, img_width=640/2, img_height=430/2, title=f"GT + Anchors after geometry pre-filter", line_pred=1.7, save_path=save_path3)
            img_counter += 1

        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)         #compute iou between gt and each anchor candidate.
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)                                 

        #conf_term = (1.0 - obj_preds_.sigmoid().detach())

        # show the iou of best anchors and gt boxes
        """for i in range(gt_bboxes_per_image.shape[0]):
            gt = gt_bboxes_per_image[i]
            ious = pair_wise_ious[i]  # IoUs of this GT with all anchors
            if ious.numel() == 0:
                print(f"[DEBUG] GT {i}: bbox={gt.tolist()} has no anchors after filtering")
                max_iou = 0.0
                max_idx = -1
                topk_ious = []
            else:
                max_iou, max_idx = ious.max(0)
                k = min(5, ious.numel())
                topk_ious = ious.topk(k).values.tolist()
            print(f"[DEBUG] GT {i}: bbox={gt.tolist()}")
            print(f"        Max IoU with any anchor: {max_iou.item():.4f} at anchor idx {max_idx.item()}")
            print(f"        IoUs with top 5 anchors: {topk_ious}")"""

        if mode == "cpu":
            obj_preds_ = obj_preds_.cpu()

        cost = (
            # 0* conf_term 
            + 3 * pair_wise_ious_loss                           #Before cls was influencing the cost. original dsec: 3* iou but changing weight not big impact(on dsec events)
            + float(1e6) * (~geometry_relation)                   #disqualify anchors that are gemoetrically invalid / far away from gt center 
        )

        (   num_fg,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, num_gt, fg_mask)                         #simota selects positve anchors for each gt
        del cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Early pre filter, retrunr several anchors that are related to a gt.
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 3
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        """print("\n[DEBUG] Geometry Constraint:")
        print("  Num GTs:", gt_bboxes_per_image.shape[0])
        print("  Num anchors:", x_centers_per_image.shape[1])
        print("  GT bboxes (first):", gt_bboxes_per_image[:1].tolist())
        print("  Example anchor centers (first 5):", 
            list(zip(x_centers_per_image[0, :5].tolist(), 
                    y_centers_per_image[0, :5].tolist())))
        print("  Anchor filter count:", anchor_filter.sum().item())
        print("  Geometry relation shape:", geometry_relation.shape)"""

        return anchor_filter, geometry_relation
    
    def simota_matching(self, cost, pair_wise_ious, num_gt, fg_mask):
        """ which anchors are matched to which gt boxes during training
        anchor: candidate predicted boxes
        positive anchor: best anchor (determined by cost) relative to the GT.
        cost: used for RANKING of anchors for each gt
        pair_wise_ious: used to decide HOW MANY anchors each gt will select
                take dynamic_k from cost → basically the same as from iou(just rescaled). But before cls was influencing cost, thats why iou was used to set the budget
        """

        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        #seleting the amount of candidates - not necessariy 10!!!
        n_candidate_k = min(20, pair_wise_ious.size(1))                                      #for each gt take top 10 ious
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)                      #sum the iou values -> determines the number of pos anchors
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)                             #clamp ensures that gt with low iou get at least one pos anchor (in represents the num of anchors not which one)

        #rank anchors for each gt to select best one
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(                                        #for each gt pick the dynamic_k anchors  with lowest cost
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1                            #matching_matrix: each row: which anchors are assigned to that gt. each anchor at most assigned to one gt
                                                                            # --> pos samples; get further supervised with regression and objectness
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)                             #sum how many gts this anchor is assigned to 
        # deal with the case that one anchor matches multiple ground-truths     --> pick the GT with lowest cost.
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        fg_mask_inboxes = anchor_matching_gt > 0                                #boolean mask of anchors assigned to any GT
        num_fg = fg_mask_inboxes.sum().item()                                   #num of all pos anchors

        fg_mask[fg_mask.clone()] = fg_mask_inboxes                              #update global mask

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)         #GT index for each positive anchors -> keep, also used for l1

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[        #predicted ious for all pos anchors
            fg_mask_inboxes
        ]
        return num_fg, pred_ious_this_matching, matched_gt_inds                 #returning the num of positive anchors, their ious and the index of the matching gt :)


    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

        for k, (reg_conv, stride_this_level, x) in enumerate(
            zip(self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            reg_x = x

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image,obj_pred_per_image, expanded_strides, x_shifts,
                    y_shifts, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")
