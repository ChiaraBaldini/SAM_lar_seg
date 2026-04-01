# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

device = "cuda:0"
medsam_model = sam_model_registry["vit_b"](checkpoint="/work/cbaldini/medSAM/code/MedSAM/work_dir/SAM-ViT-B-20250728-1456/sam_freezedENC_model_laryngoscope_best.pth") #/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250401-1801/medsam_model_best_converted.pth") #was "/work/cbaldini/medSAM/code/MedSAM/medsam_vit_b.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()

img_path="/work/cbaldini/medSAM/code/laryngoscope_dataset/test/images"
label_path="/work/cbaldini/medSAM/code/laryngoscope_dataset/test/masks"
seg_path="/work/cbaldini/medSAM/code/MedSAM/assets/Laryngoscope_SAM_freezedENC_test_bbimage"
if not os.path.exists(seg_path):
    os.mkdir(seg_path)

for i in os.listdir(img_path):
    if 'frame' in i:
        img_np = io.imread(join(img_path,i))
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # txt=join(label_path,i.split('.')[0]+'.txt')
        # with open(os.path.join(txt)) as fh:
        #     for line in fh:
        #         classe, coordinate = line.strip().split(None, 1)
        #         coordinate=coordinate.split(' ')
        #         GTx_min=(float(coordinate[0])*W-float(coordinate[2])*W/2)
        #         GTy_min=(float(coordinate[1])*H-float(coordinate[3])*H/2)
        #         GTx_max=(float(coordinate[0])*W+float(coordinate[2])*W/2)
        #         GTy_max=(float(coordinate[1])*H+float(coordinate[3])*H/2)
        # box_np=np.array([[int(GTx_min),int(GTy_min),int(GTx_max),int(GTy_max)]])
        # box_1024 = box_np / np.array([W, H, W, H]) * 1024
        # with torch.no_grad():
        #     image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

        # medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        # # io.imsave(
        # #     join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
        # #     medsam_seg,
        # #     check_contrast=False,
        # # )

        # # # %% visualize results
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(img_3c)
        # show_box(box_np[0], ax[0])
        # ax[0].set_title("Input Image and Bounding Box")
        # ax[1].imshow(img_3c)
        # show_mask(medsam_seg, ax[1])
        # show_box(box_np[0], ax[1])
        # ax[1].set_title("MedSAM Segmentation")
        # plt.savefig(os.path.join(seg_path, "seg_" + i))
        # # # plt.show()

        # txt = join(label_path, i.split('.')[0] + '.txt')
        # boxes = []  # Store multiple boxes
        # with open(txt, 'r') as fh:
        #     for line in fh:
        #         classe, coordinate = line.strip().split(None, 1)
        #         coordinate = coordinate.split(' ')
        #         GTx_min = (float(coordinate[0]) * W - float(coordinate[2]) * W / 2)
        #         GTy_min = (float(coordinate[1]) * H - float(coordinate[3]) * H / 2)
        #         GTx_max = (float(coordinate[0]) * W + float(coordinate[2]) * W / 2)
        #         GTy_max = (float(coordinate[1]) * H + float(coordinate[3]) * H / 2)
        #         boxes.append([int(GTx_min), int(GTy_min), int(GTx_max), int(GTy_max)])

        mask_path = join(label_path, i.split('.')[0] + '.jpg')  # o .jpg, a seconda del formato
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        ys, xs = np.where(mask > 0)
        boxes = []
        if len(xs) > 0 and len(ys) > 0:
            # x_min, x_max = xs.min(), xs.max()
            # y_min, y_max = ys.min(), ys.max()
            # # Perturbed
            # x_min, x_max = xs.min()-20, xs.max()+20
            # y_min, y_max = ys.min()-20, ys.max()+20
            x_min, x_max = 0, W
            y_min, y_max = 0, H
            boxes.append([x_min, y_min, x_max, y_max])

        box_np = np.array(boxes)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024  # Normalize for MedSAM

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

        # Visualization with all bounding boxes and separate masks
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)

        # Show all bounding boxes on the original image
        for box in box_np:
            show_box(box, ax[0])  
        ax[0].set_title("Input Image and Bounding Boxes all image")

        ax[1].imshow(img_3c)

        # Process each bounding box separately
        box_original = (box_1024 * np.array([W, H, W, H]) / 1024).astype(int)
        for box in box_1024:
            medsam_seg = medsam_inference(medsam_model, image_embedding, box[None, :], H, W)
            show_mask(medsam_seg, ax[1])  # Overlay each mask separately
        for box in box_original:
            show_box(box, ax[1])

        ax[1].set_title("SAM Segmentation after tuning")

        plt.savefig(os.path.join(seg_path, "seg_" + i))

