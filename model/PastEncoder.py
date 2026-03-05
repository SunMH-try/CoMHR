import torch
from torch import nn
from torch.nn import functional as F
from utils.imutils import cam_crop2full, vis_img
from .HumanGroupNet import MS_HGNN_oridinary,MS_HGNN_hyper
from collections import namedtuple
from utils.geometry import perspective_projection
from utils.rotation_conversions import *
import cv2
from model.backbones.resnet50 import ResNet50


args = namedtuple('args', [
    'hidden_dim',
    'hyper_scales',
    'learn_prior',
    'nmp_layers',
])

class PastEncoder(nn.Module):
    def __init__(self, args, in_dim=2048):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.scale_number = len(args.hyper_scales)
        self.nmp_layers =args.nmp_layers
        self.features_project = nn.Linear(1024, 256)

      
        self.interaction = MS_HGNN_oridinary(
            embedding_dim=2048,
            h_dim=self.model_dim,
            mlp_dim=2048,
            bottleneck_dim=self.model_dim,
            batch_norm=0,
            nmp_layers=self.nmp_layers
        )

        if len(args.hyper_scales) > 0:
            self.interaction_hyper = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[0]
            )
        if len(args.hyper_scales) > 1:
            self.interaction_hyper2 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[1]
            )

        if len(args.hyper_scales) > 2:
            self.interaction_hyper3 = MS_HGNN_hyper(
                embedding_dim=self.model_dim,
                h_dim=self.model_dim,
                mlp_dim=64,
                bottleneck_dim=self.model_dim,
                batch_norm=0,
                nmp_layers=self.nmp_layers,
                scale=args.hyper_scales[2]
            )
        self.tz_project = nn.Linear(257, 256)
        self.tz_norm = nn.LayerNorm(256)

    def add_category(self,x):
        B = x.shape[0]
        N = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.repeat(B,1,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def convert_color(self, gray):
        im_color = cv2.applyColorMap(cv2.convertScaleAbs(gray, alpha=1),cv2.COLORMAP_JET)
        return im_color

    def viz_two_affinity(self, collectives, corrs):
        
        collectives = collectives.detach().cpu().numpy()
        corrs = corrs.detach().cpu().numpy()
        for collective, corr in zip(collectives, corrs):
            ratiox = 800/int(collective.shape[0])
            ratioy = 800/int(collective.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            collective = self.convert_color(collective*255)
            corr = self.convert_color(corr*255)
            # im = cv2.resize(im, dsize=None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow('collective',0)
            cv2.resizeWindow('collective',int(collective.shape[1]*ratio),int(collective.shape[0]*ratio))
            cv2.imshow('collective',collective)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(corr.shape[1]*ratio),int(corr.shape[0]*ratio))
            cv2.imshow('affinity',corr)
            cv2.waitKey()

    def viz_affinity(self, aff_map):
        viz = []
        aff_maps = aff_map.detach().cpu().numpy()
        for im in aff_maps:
            ratiox = 800/int(im.shape[0])
            ratioy = 800/int(im.shape[1])
            if ratiox < ratioy:
                ratio = ratiox
            else:
                ratio = ratioy
        
            im = self.convert_color(im*255)
            cv2.namedWindow('affinity',0)
            cv2.resizeWindow('affinity',int(im.shape[1]*ratio),int(im.shape[0]*ratio))
            cv2.imshow('affinity',im)
            cv2.waitKey()
            viz.append(im)
        return viz

    def forward(self, inputs, features_inputs, Tz, batch_size, agent_num, mask):
        features_inputs = self.features_project(features_inputs.view(batch_size*agent_num, -1))
        features_inputs = features_inputs.view(batch_size, agent_num, -1)

        length = inputs.shape[1]
        if self.training:
            # drop_prob = 0.2
            # random_mask = (torch.rand_like(mask) > drop_prob).float()
            # mask = mask * random_mask
            mask = mask
        else:
            mask = mask        

        inputs = inputs * mask[:, None]
        
        ftraj_input = inputs.view(batch_size, agent_num, -1)

        if Tz is not None:
            Tz = Tz.view(batch_size, agent_num, 1)               
            ftraj_input = torch.cat([ftraj_input, Tz], dim=-1)    
            ftraj_input = F.relu(self.tz_project(ftraj_input))   
            ftraj_input = self.tz_norm(ftraj_input)               

        mask = mask.view(batch_size, agent_num)
        mask = torch.matmul(mask[:, :, None], mask[:, None, :])

        query_input = F.normalize(ftraj_input, p=2, dim=2)
        feat_corr = torch.matmul(query_input, query_input.permute(0, 2, 1))

        viz_affinity = False
        if viz_affinity:
            aff_maps = self.viz_affinity(feat_corr)

        # interaction
        ftraj_inter, _ = self.interaction(ftraj_input, mask)

        if len(self.args.hyper_scales) > 0:
            ftraj_inter_hyper, _ = self.interaction_hyper(features_inputs, feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 1:
            ftraj_inter_hyper2, _ = self.interaction_hyper2(features_inputs, feat_corr, mask, viz=False)
        if len(self.args.hyper_scales) > 2:
            ftraj_inter_hyper3, _ = self.interaction_hyper3(features_inputs, feat_corr, mask)


        if len(self.args.hyper_scales) == 0:
            final_feature = torch.cat((ftraj_input, ftraj_inter), dim=-1)
        elif len(self.args.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper), dim=-1)
        elif len(self.args.hyper_scales) == 2:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2), dim=-1)
        elif len(self.args.hyper_scales) == 3:
            final_feature = torch.cat((ftraj_input, ftraj_inter, ftraj_inter_hyper, ftraj_inter_hyper2, ftraj_inter_hyper3), dim=-1)

        output_feature = final_feature.view(batch_size * agent_num, -1)

        return output_feature
