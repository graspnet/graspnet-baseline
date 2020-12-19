import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Viewpoint estimation module from seed point features.

        Args:
            num_view: int
                number of views generated from each each seed point
            seed_feature_dim:
                numbe of channels of seed point features
        """
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2+self.num_view)

    def forward(self, seed_xyz, seed_features, end_points):
        """ Forward pass

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
            end_points: dict
        Returns:
            objectness_score:
            view_score:
            vp_xyz: (batch_size, num_seed, 3) top viewpoint xyz on spherical surface
            vp_rot: (batch_size, num_seed, 3, 3) rotation matrix based on top viewpoints
        """
        B, num_seed, _ = seed_xyz.size()
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        objectness_score = features[:, :2, :] # (B, 2, num_seed)
        view_score = features[:, 2:2+self.num_view, :].transpose(1,2).contiguous() # (B, num_seed, num_view)
        end_points['objectness_score'] = objectness_score
        end_points['view_score'] = view_score

        # print(view_score.min(), view_score.max(), view_score.mean())
        top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed)
        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        template_views = generate_grasp_views(self.num_view).to(features.device) # (num_view, 3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3)
        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot

        return end_points


class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]
        
        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot):
        """
        seed_xyz: (batch_size, num_sample, 3)
        pointcloud: (batch_size, num_seed, 3)
        vp_rot: (batch_size, num_seed, 3, 3)
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            )) # (batch_size, feature_dim, num_seed, nsample)
        grouped_features = torch.stack(grouped_features, dim=3) # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed*num_depth, self.nsample) # (batch_size, feature_dim, num_seed*num_depth, nsample)

        vp_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features

        
class OperationNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3*num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2*self.num_angle:3*self.num_angle]
        return end_points

    
class ToleranceNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        # tolerance (num_angle)
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points