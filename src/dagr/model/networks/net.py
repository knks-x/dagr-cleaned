import torch

import torch_geometric.transforms as T

from torch_geometric.data import Data
from dagr.model.layers.ev_tgn import EV_TGN
from dagr.model.layers.pooling import Pooling
from dagr.model.layers.conv import Layer
from dagr.model.layers.components import Cartesian
from dagr.model.networks.net_img import HookModule
from dagr.model.utils import shallow_copy
from torchvision.models import resnet18, resnet34, resnet50


def sampling_skip(data, image_feat):
    image_feat_at_nodes = sample_features(data, image_feat)
    return torch.cat((data.x, image_feat_at_nodes), dim=1)

def compute_pooling_at_each_layer(pooling_dim_at_output, num_layers):
    py, px = map(int, pooling_dim_at_output.split("x"))
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    poolings = []
    for i in range(num_layers):
        pooling = pooling_base / 2 ** (3 - i)
        pooling[-1] = 1
        poolings.append(pooling)
    poolings = torch.stack(poolings)
    return poolings


class Net(torch.nn.Module):
    def __init__(self, args, height, width):
        super().__init__()                                                                              #use parent (nn.module) to initialize the object

        channels = [1, int(args.base_width*32), int(args.after_pool_width*64),                          #list of channels sizes
                    int(args.net_stem_width*128),                                                       #6 channel sizes for 5layers: [1, 16, 64, 64, 64] for dagr-s
                                                                                                        #1: input channel for event data(single-channel event input)
                    int(args.net_stem_width*128),
                    int(args.net_stem_width*128)]

        self.out_channels_cnn = []

        self.use_image = args.use_image
        self.num_scales = args.num_scales                                                               #how many detection scales backbone/head should produce; scales set (8,16,32)

        self.num_classes=1                                                                              #no cls but coco eval needs a class 

        self.events_to_graph = EV_TGN(args)                                                             #converts event streams into graph representation
        
        output_channels = channels[1:]                                                                  # drops the initial 1 and keeps rest of the channel integers
                                                                                                        #output channels: [16, 64, 64, 64, 64]
        self.out_channels = output_channels[-2:]                                                        #last two entries: channel sizes of the final feature maps

        input_channels = channels[:-1]                                                                  #input channels: [1, 16, 64, 64, 64]

        # parse x and y pooling dimensions at output
        poolings = compute_pooling_at_each_layer(args.pooling_dim_at_output, num_layers=4)              #helper functions for pooling/downsampling for each layer
        max_vals_for_cartesian = 2*poolings[:,:2].max(-1).values                                        #value that isused to scale/normalize coordinates in Cartesian transforms
        
        #use largest strides (16, 32)
        self.strides = torch.ceil(poolings[-2:,1] * height).numpy().astype("int32").tolist()            
        self.strides = self.strides[-self.num_scales:]                                                  #set num of strides to num_scales = 2 -> head will output two prediction maps (with stride 16, 32)

        effective_radius = 2*float(int(args.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)                   #computes edge attributes 

        self.conv_block1 = Layer(2+input_channels[0], output_channels[0], args=args)                    #2+ : likely adding two coordinate channels alongside numeric features

        cart1 = T.Cartesian(norm=True, cat=False, max_value=2*effective_radius)
        self.pool1 = Pooling(poolings[0], width=width, height=height, batch_size=args.batch_size,
                             transform=cart1, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer2 = Layer(input_channels[1]+2, output_channels[1], args=args)

        cart2 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[1])
        self.pool2 = Pooling(poolings[1], width=width, height=height, batch_size=args.batch_size,
                             transform=cart2, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer3 = Layer(input_channels[2]+2, output_channels[2],  args=args)

        cart3 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[2])
        self.pool3 = Pooling(poolings[2], width=width, height=height, batch_size=args.batch_size,
                             transform=cart3, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer4 = Layer(input_channels[3]+2, output_channels[3],  args=args)

        cart4 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[3])
        self.pool4 = Pooling(poolings[3], width=width, height=height, batch_size=args.batch_size,
                             transform=cart4, aggr='mean', keep_temporal_ordering=args.keep_temporal_ordering)

        self.layer5 = Layer(input_channels[4]+2, output_channels[4],  args=args)

        self.cache = []

    def get_output_sizes(self):
        poolings = [self.pool3.voxel_size[:2], self.pool4.voxel_size[:2]]
        output_sizes = [(1 / p + 1e-3).cpu().int().numpy().tolist()[::-1] for p in poolings]
        return output_sizes

    def forward(self, data: Data, reset=True):

        if hasattr(data, 'reset'):
            reset = data.reset

        data = self.events_to_graph(data, reset=reset)                              #return events

        data = self.edge_attrs(data)                                                #attach edge attributes
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.conv_block1(data)                                               #first layer
        data = self.pool1(data)                                                     #first pooling

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer2(data)
        data = self.pool2(data)

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer3(data)
        data = self.pool3(data)

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer4(data)

        out3 = shallow_copy(data)                                                   #first "output" before pool4 -> to detection head like this
        out3.pooling = self.pool3.voxel_size[:3]

        data = self.pool4(data)

        rel_delta = data.pos[:,:2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer5(data)

        out4 = data                                                                #second "output" from last layer -> to detection head like this
        out4.pooling = self.pool4.voxel_size[:3]

        output = [out3, out4]

        return output[-self.num_scales:]


def sample_features(data, image_feat, image_sample_mode="bilinear"):
    if data.batch is None or len(data.batch) != len(data.pos):
        data.batch = torch.zeros(len(data.pos), dtype=torch.long, device=data.x.device)
    return _sample_features(data.pos[:,0] * data.width[0],
                            data.pos[:,1] * data.height[0],
                            data.batch.float(), image_feat,
                            data.width[0],
                            data.height[0],
                            image_feat.shape[0],
                            image_sample_mode)

def _sample_features(x, y, b, image_feat, width, height, batch_size, image_sample_mode):
    x = 2 * x / (width - 1) - 1
    y = 2 * y / (height - 1) - 1

    batch_size = batch_size if batch_size > 1 else 2
    b = 2 * b / (batch_size - 1) - 1

    grid = torch.stack((x, y, b), dim=-1).view(1, 1, 1,-1, 3) # N x D_out x H_out x W_out x 3 (N=1, D_out=1, H_out=1)
    image_feat = image_feat.permute(1,0,2,3).unsqueeze(0) # N x C x D x H x W (N=1)

    image_feat_sampled = torch.nn.functional.grid_sample(image_feat,
                                                         grid=grid,
                                                         mode=image_sample_mode,
                                                         align_corners=True) # N x C x H_out x W_out (H_out=1, N=1)

    image_feat_sampled = image_feat_sampled.view(image_feat.shape[1], -1).t()

    return image_feat_sampled




