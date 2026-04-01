# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from re import X
# from tkinter import N
from tokenize import Double
import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d, softmax_one
from einops import rearrange

def mask_to_bbox(mask):
    """Extracts the bounding box [x_min, y_min, x_max, y_max] from a binary mask."""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array([x_min, y_min, x_max, y_max])

def salient_points_to_bbox(mask, num_points=100):
    """
    Extracts the most salient points from the mask and computes the bounding box that contains them.
    """
    pts, _ = improved_pos_neg_clicks(mask, pos_prompt_number=num_points, neg_prompt_number=0)
    if len(pts) == 0:
        h, w = mask.shape
        return np.array([0, 0, w-1, h-1])
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    return np.array([x_min, y_min, x_max, y_max])

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class vitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx):
        q = self.to_q(qx)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(kx).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn =  softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2in(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x1, x2):
        for attn, ff in self.layers:
            ax = attn(x1, x2)
            x1 = ax + x1
            x1 = ff(x1) + x1
        return x1


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x1, x2):
        for attn1, attn2, ff1, ff2 in self.layers:
            ax1, ax2 = attn1(x1, x2), attn2(x2, x1)
            x1, x2 = ax1 + x1, ax2 + x2
            x1 = ff1(x1) + x1
            x2 = ff2(x2) + x2
        return x1, x2


class Prompt_Embedding_Generator(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        base_dim: int=48,
        num_heads: int = 8,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = out_dim
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.scale = (out_dim//self.num_heads)**-0.5

        self.object_token = nn.Parameter(torch.randn(1, 50, self.embed_dim))
        self.cross_token_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_image_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)
        
    def forward(self,
        img_embedding: torch.Tensor,
        output_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returning new_img_embedding, new_output_token, and object_token

        Arguments:
          img_embedding: torch.Tensor with shape (B, embed_dim, 32, 32)
          output_token: torch.Tensor with shape (B, 5, 256)

        Returns:
          torch.Tensor: new img embedding, with shape (B, embed_dim, 16, 16).
          torch.Tensor: new ouput token, with shape (B, 5, 256).
          torch.Tensor: object token, with shape Bx1x(embed_dim).
        """
        #img_embedding = self.feature_adapter(img_embedding)
        b, c, h, w = img_embedding.shape
        img_embedding = rearrange(img_embedding, 'b c h w -> b (h w) c')
        object_token, new_output_token = self.cross_token_token(self.object_token, output_token)
        object_token = self.token_proj(object_token) + self.object_token
        new_output_token = self.token_proj(new_output_token) + output_token
        tokens = torch.cat([object_token, output_token], dim=1) # [b 6 d]
        new_img_embedding, tokens = self.cross_image_token(img_embedding, tokens) 
        new_img_embedding = rearrange(new_img_embedding, 'b (h w) c -> b c h w', h=h)
        return new_img_embedding, tokens[:, :1, :], tokens[:, 1:, :]


class MaskAttention(nn.Module):
    def __init__(self, embedding=256, kernel_size=7):
        super(MaskAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(embedding, embedding, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(embedding, embedding, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(embedding, 1, kernel_size=3, padding=1, bias=False)
        self.convup1 = nn.Conv2d(1, embedding//2, kernel_size=3, padding=1, bias=False)
        self.convup2 = nn.Conv2d(embedding//2, embedding, kernel_size=3, padding=1, bias=False)
 
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x1)
        x3 = self.conv3(x1)
    
        avg_attn = torch.mean(x2, dim=1, keepdim=True)
        max_attn, _ = torch.max(x2, dim=1, keepdim=True)
        attn = torch.cat([avg_attn, max_attn, x3], dim=1)
        attn = self.conv(attn)
        attn1 = self.sigmoid(attn)
        x_up = self.convup1(attn)
        x_up = self.convup2(x_up)
        x_up = attn1 * x_up

        return x_up + x1, attn


# def pos_neg_clicks(mask, class_id=1, pos_prompt_number=1, neg_prompt_number=1):
#     pos_indices = np.argwhere(mask == class_id)
#     pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
#     pos_label = 1
#     if len(pos_indices) == 0:
#         pos_label = -1 # or 0
#         pos_indices = np.argwhere(mask != class_id)
#         pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
#     pos_num = min(len(pos_indices), pos_prompt_number)
#     pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_num)
#     pos_prompt = pos_indices[pos_prompt_indices]
#     pos_label = np.repeat(pos_label, pos_num)

#     neg_indices = np.argwhere(mask != class_id)
#     neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
#     neg_num = pos_prompt_number + neg_prompt_number - pos_num
#     neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_num)
#     neg_prompt = neg_indices[neg_prompt_indices]
#     neg_label = np.repeat(0, neg_num)

#     pt = np.vstack((pos_prompt, neg_prompt))
#     point_label = np.hstack((pos_label, neg_label))
#     return pt, np.array(point_label)

# def make_prompt_from_mask(mask):
#     pts, point_labels = [], []
#     with torch.no_grad():
#         predict = torch.sigmoid(mask)
#         predict = predict.detach().cpu().numpy()  
#         seg = predict[:, 0, :, :] > 0.5 
#         for i in range(seg.shape[0]):
#             pt, point_label = pos_neg_clicks(seg[i, :, :], pos_prompt_number=10, neg_prompt_number=0)
#             pts.append(pt[None, :, :])
#             point_labels.append(point_label[None, :])
#         pts = np.concatenate(pts, axis=0)
#         point_labels = np.concatenate(point_labels, axis=0)
#     coords_torch = torch.as_tensor(pts, dtype=torch.float32, device=mask.device)
#     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=mask.device)
#     if len(pts.shape) == 2:
#         coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
#     pts = (coords_torch, labels_torch)
#     return pts

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from sklearn.cluster import KMeans

def improved_pos_neg_clicks(mask, class_id=1, pos_prompt_number=1, neg_prompt_number=1):
    pos_indices = np.argwhere(mask == class_id)
    neg_indices = np.argwhere(mask != class_id)
    
    if len(pos_indices) == 0:
        pos_label = -1
        pos_indices = neg_indices
        neg_indices = np.argwhere(mask == class_id)
    else:
        pos_label = 1
    
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    
    def select_smart_points(indices, mask_binary, num_points, strategy="combined"):
        if len(indices) == 0:
            return np.array([]).reshape(0, 2)
        
        if len(indices) <= num_points:
            return indices
        
        if strategy == "centroids":
            # STRATEGY 1: Centroids of connected components
            try:
                labeled_mask = measure.label(mask_binary)
                regions = measure.regionprops(labeled_mask)
                
                centroids = []
                for region in regions:
                    centroid = region.centroid
                    centroids.append([int(centroid[1]), int(centroid[0])])  # (x, y)
                
                if centroids:
                    centroids = np.array(centroids)
                    return centroids[:num_points]
            except:
                pass
        
        elif strategy == "max_distance":
            # STRATEGY 2: Maximum Euclidean distance
            try:
                selected = [indices[0]]
                
                for _ in range(min(num_points - 1, len(indices) - 1)):
                    distances = cdist(indices, selected, metric='euclidean')
                    min_distances = np.min(distances, axis=1)
                    farthest_idx = np.argmax(min_distances)
                    selected.append(indices[farthest_idx])
                    
                return np.array(selected)
            except:
                pass
        
        elif strategy == "kmeans":
            # STRATEGY 3: K-means clustering
            try:
                kmeans = KMeans(n_clusters=min(num_points, len(indices)), 
                              random_state=42, n_init=10)
                clusters = kmeans.fit(indices)
                
                selected = []
                for center in clusters.cluster_centers_:
                    distances = np.linalg.norm(indices - center, axis=1)
                    closest_idx = np.argmin(distances)
                    selected.append(indices[closest_idx])
                
                return np.array(selected)
            except:
                pass
        
        elif strategy == "combined":
            # COMBINED STRATEGY: Center + Edges + Distribution
            try:
                # 1. Center of mass
                center_y, center_x = np.mean(indices, axis=0)
                center = np.array([[int(center_x), int(center_y)]])
                
                if num_points == 1:
                    return center
                
                # 2. Points on the edges (using erosion)
                kernel = np.ones((3,3), np.uint8)
                eroded = cv2.erode(mask_binary.astype(np.uint8), kernel, iterations=1)
                edge_mask = mask_binary.astype(np.uint8) - eroded
                edge_indices = np.argwhere(edge_mask > 0)
                
                if len(edge_indices) > 0:
                    edge_indices[:, [0,1]] = edge_indices[:, [1,0]]  # (x, y)
                    # Select edge points that are farthest from the center
                    distances = cdist(edge_indices, center, metric='euclidean').flatten()
                    sorted_indices = np.argsort(distances)[::-1]
                    
                    # Take half of the points from the edges
                    edge_points_num = min(num_points // 2, len(edge_indices))
                    edge_points = edge_indices[sorted_indices[:edge_points_num]]
                    
                    # 3. Complete with uniform distribution over the remaining points
                    remaining_points = num_points - 1 - edge_points_num
                    if remaining_points > 0:
                        # Use K-means on internal points
                        internal_indices = []
                        for idx in indices:
                            x, y = idx
                            if mask_binary[y, x] == 1 and edge_mask[y, x] == 0:
                                internal_indices.append(idx)
                        
                        if len(internal_indices) > 0:
                            internal_indices = np.array(internal_indices)
                            if len(internal_indices) <= remaining_points:
                                internal_points = internal_indices
                            else:
                                # K-means
                                kmeans = KMeans(n_clusters=remaining_points, 
                                              random_state=42, n_init=10)
                                clusters = kmeans.fit(internal_indices)
                                internal_points = []
                                for center_km in clusters.cluster_centers_:
                                    distances = np.linalg.norm(internal_indices - center_km, axis=1)
                                    closest_idx = np.argmin(distances)
                                    internal_points.append(internal_indices[closest_idx])
                                internal_points = np.array(internal_points)
                        else:
                            internal_points = np.array([]).reshape(0, 2)
                    else:
                        internal_points = np.array([]).reshape(0, 2)
                    
                    all_points = [center]
                    if len(edge_points) > 0:
                        all_points.append(edge_points)
                    if len(internal_points) > 0:
                        all_points.append(internal_points)
                    
                    return np.vstack(all_points)
                
            except Exception as e:
                print(f"Error durin combined strategy: {e}")
                pass
        
        
        try:
            selected = [indices[0]]
            min_distance = max(5, len(indices) // 20)
            
            for _ in range(min(num_points - 1, len(indices) - 1)):
                candidates = []
                for i, point in enumerate(indices):
                    # Check that the point is far enough from already selected points
                    if all(np.linalg.norm(point - sel_point) >= min_distance 
                           for sel_point in selected):
                        candidates.append(point)
                
                if candidates:
                    selected.append(candidates[np.random.randint(len(candidates))])
                else:
                    # If there are no candidates, pick the farthest point
                    distances = cdist(indices, selected, metric='euclidean')
                    min_distances = np.min(distances, axis=1)
                    farthest_idx = np.argmax(min_distances)
                    selected.append(indices[farthest_idx])
            
            return np.array(selected)
        except:
            random_indices = np.random.choice(len(indices), 
                                            size=min(num_points, len(indices)), 
                                            replace=False)
            return indices[random_indices]

    mask_binary = (mask == class_id).astype(np.uint8)
    
    # Select positive points using combined strategy
    if pos_prompt_number > 0:
        pos_prompt = select_smart_points(pos_indices, mask_binary, 
                                       min(len(pos_indices), pos_prompt_number), 
                                       strategy="combined")
    else:
        pos_prompt = np.array([]).reshape(0, 2)
    
    # Select negative points with uniform distribution
    neg_num = pos_prompt_number + neg_prompt_number - len(pos_prompt)
    if neg_num > 0 and len(neg_indices) > 0:
        neg_mask_binary = (mask != class_id).astype(np.uint8)
        neg_prompt = select_smart_points(neg_indices, neg_mask_binary, 
                                       min(len(neg_indices), neg_num), 
                                       strategy="max_distance")
    else:
        neg_prompt = np.array([]).reshape(0, 2)

    # Combine points and labels
    if len(pos_prompt) > 0 and len(neg_prompt) > 0:
        point_coords = np.vstack([pos_prompt, neg_prompt])
        point_labels = np.concatenate([
            np.full(len(pos_prompt), pos_label),
            np.zeros(len(neg_prompt))
        ])
    elif len(pos_prompt) > 0:
        point_coords = pos_prompt
        point_labels = np.full(len(pos_prompt), pos_label)
    elif len(neg_prompt) > 0:
        point_coords = neg_prompt
        point_labels = np.zeros(len(neg_prompt))
    else:
        point_coords = np.array([]).reshape(0, 2)
        point_labels = np.array([])
    
    return point_coords, point_labels

def make_prompt_from_mask(mask):
    with torch.no_grad():
        predict = torch.sigmoid(mask)
        seg = predict[:, 0, :, :] > 0.5
        
        # pts_list = []
        # pts_label_list = []
        
        # for i in range(seg.shape[0]):
        #     # Use the improved function instead of pos_neg_clicks
        #     pt, point_label = improved_pos_neg_clicks(
        #         seg[i].cpu().numpy(), 
        #         pos_prompt_number=10,  # More points for better coverage
        #         neg_prompt_number=0
        #     )
            
        #     if len(pt) > 0:
        #         pts_list.append(torch.from_numpy(pt))
        #         pts_label_list.append(torch.from_numpy(point_label))
        #     else:
        #         # Fallback: central point of the image
        #         h, w = seg[i].shape
        #         center_pt = np.array([[w//2, h//2]])
        #         pts_list.append(torch.from_numpy(center_pt))
        #         pts_label_list.append(torch.ones(1))
        
        # if pts_list:
        #     pts = torch.stack(pts_list, dim=0).float()
        #     pts_label = torch.stack(pts_label_list, dim=0).float()
        # else:
        #     # Final fallback 
        #     batch_size = seg.shape[0]
        #     pts = torch.zeros(batch_size, 1, 2)
        #     pts_label = torch.zeros(batch_size, 1)
            
        # return pts

        with torch.no_grad():
        predict = torch.sigmoid(mask)
        seg = predict[:, 0, :, :] > 0.5
        
        bboxes = []
        for i in range(seg.shape[0]):
            bbox = salient_points_to_bbox(seg[i].cpu().numpy(), num_points=num_points)
            bboxes.append(torch.from_numpy(bbox).float())
        return torch.stack(bboxes, dim=0)
