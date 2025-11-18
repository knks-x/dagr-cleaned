import torch

import torch.nn.functional as F

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net import Net
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format


class DAGR(YOLOX):                                                              #dagr inherits from yolox, yolox is parent class
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height                                             #scale because dataset width/height is unscaled bc downscaling of width height in data.height not dataset.height
        self.width = width 

        backbone = Net(args, height=height, width=width)                        #construct object backbone of class Net --> go to Net
        head = GNNHead(num_classes=backbone.num_classes,
                       in_channels=backbone.out_channels,
                       in_channels_cnn=backbone.out_channels_cnn,
                       strides=backbone.strides,
                       pretrain_cnn=args.pretrain_cnn,
                       args=args)

        super().__init__(backbone=backbone, head=head)                          #call yolox constructor (parent) to initialize these parts

    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width+1)
        self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

        rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
        self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
        self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
        self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
        self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if self.head.num_scales > 1:
            self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            #bbox-> prediction at image_1; bbox0-> prediciton at image_0!
            targets = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
    
            outputs = YOLOX.forward(self, x, targets)                         
            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)                                     #this is returned when model(data) is called in train mode

        #detections after postprocess: of format dict with tensor lists: "boxes": detections[x1, y1, x2, y2], "scores": detections[objectness], "labels": detections[0]
        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        #targets after convert in format: "boxes": bbox[:,:4], "labels": bbox[:, 4]
        #use bbox0 for front 
        if return_targets and hasattr(x, 'bbox0'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret                                                           #this is returned when model(data) is called in eval mode

"""
class CNNHead(YOLOXHead):
    def forward(self, xin):
        outputs = dict(reg_output=[], obj_output=[])

        for k, (reg_conv, x) in enumerate(zip(self.reg_convs, xin)):
            x = self.stems[k](x)
            #cls_x = x
            reg_x = x

            #cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            #outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs
"""

class GNNHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=[256, 512, 1024],
        act="silu",
        depthwise=False,
        pretrain_cnn=False,
        args=None
    ):
        YOLOXHead.__init__(self, num_classes, args.yolo_stem_width, strides, in_channels, act, depthwise)

        self.pretrain_cnn = pretrain_cnn
        self.num_scales = args.num_scales
        self.use_image = args.use_image
        self.batch_size = args.batch_size
        self.no_events = args.no_events

        self.in_channels = in_channels
        self.n_anchors = 1
        self.num_classes = num_classes

        n_reg = max(in_channels)
        self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.obj_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        #self.use_l1 = True
        #self.l1_loss = torch.nn.L1Loss(reduction="none")                                                               #already init in Yolo_head
        #self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        #self.iou_loss = IOUloss(reduction="none")
        #self.strides = strides
        #self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def process_feature(self, x, stem, reg_conv, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be foudn from the data, especially when nodes=0
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return reg_output, obj_output
    
    def process_reg_feature(self, x, stem, reg_conv, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be foudn from the data, especially when nodes=0
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        #placeholder values for confidence 
        #obj_output = torch.zeros((batch_size, 1, self.height, self.width), device=reg_output.device, dtype=reg_output.dtype)
        obj_output=torch.full_like(reg_output[:, :1, :, :], -10.0)
        
        return reg_output, obj_output


    def forward(self, xin: Data, labels=None, imgs=None):
        """
        input:
            xin= fpn outputs from backbone in yolox forward
            labels=targets, imgs=data(batch)
        output: 
            losses when training
            decoded outputs when eval (det values strided back in normal pixel space, not anchor dependent values)
        """
        hybrid_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        image_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        batch_size = len(out_cnn["reg_output"][0]) if self.use_image else self.batch_size                                    #[N, 5+C] --> N is batchsize so fits for all cls, reg, obj
        
        #first head: 
        reg_output, obj_output = self.process_feature(xin[0], self.stem1,                                                    #processes the backbone outpout(xin) through the detection head (for the first scale)
                                                      self.reg_conv1, self.reg_pred1, 
                                                      self.obj_pred1, batch_size=self.batch_size, cache=self.cache)          #returns reg(4channel) + obj(1channel) prediction 

        self.collect_outputs(reg_output, obj_output, 0, self.strides[0], ret=hybrid_out)

        #second head: remove confidence detection -> keep because its always 0 or 0.5 after sig; filter out in the 
        if self.num_scales > 1:
            reg_output, obj_output = self.process_feature(xin[1], self.stem2,                                                #processes backbone output for second scale
                                                          self.reg_conv2, self.reg_pred2,
                                                          self.obj_pred2, batch_size=batch_size, cache=self.cache)

            self.collect_outputs(reg_output, obj_output, 1, self.strides[1], ret=hybrid_out)                                 #sigmoid on obj logits in eval mode 

        if self.training:
            #debug
            #all_outputs = torch.cat(hybrid_out['outputs'], 1)  # shape [batch, n_anchors, 5?]

            return self.get_losses(                                                                                         
                imgs,
                hybrid_out['x_shifts'],
                hybrid_out['y_shifts'],
                hybrid_out['expanded_strides'],
                labels,
                torch.cat(hybrid_out['outputs'], 1),
                hybrid_out['origin_preds'],
                dtype=xin[0].x.dtype,
                events=imgs
            )
        else:
            out = image_out['outputs'] if self.no_events else hybrid_out['outputs']

            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=out[0].type())                                                        #regression logits in pixel space, scores in sigmoid

    def collect_outputs(self, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))

            if self.use_l1:
                batch_size = reg_output.shape[0]
                reg_output = reg_output.view(batch_size, 4, -1).permute(0, 2, 1)
                # shape [B, HW, 4]
                ret['origin_preds'].append(reg_output.clone())

        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs

