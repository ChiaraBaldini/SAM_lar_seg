import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from skimage import io, measure, transform
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

join = os.path.join

import argparse
from skimage.measure import label, regionprops
import numpy as np
from skimage.measure import label

def keep_largest_connected_component(mask):
    """
    Mantiene solo la componente connessa più grande in una maschera binaria.
    """
    labeled_mask = label(mask)
    if labeled_mask.max() == 0:
        return mask
    largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
    return largest_component.astype(np.uint8)

def extract_bounding_box(mask):
    labeled_mask = label(mask)  # Label connected components
    regions = regionprops(labeled_mask)

    bboxes = []
    for region in regions:
        y_min, x_min, y_max, x_max = region.bbox  # Get bounding box (row, col, row, col)
        # bboxes.append([x_min, y_min, x_max, y_max])  # Convert to (x_min, y_min, x_max, y_max)
        if (x_max-x_min)*(y_max-y_min)>100:   # to filter out small BB
            bboxes.append([x_min, y_min, x_max, y_max])  # Convert to (x_min, y_min, x_max, y_max)
    
    return bboxes

def extract_center_points(mask):
    labeled_mask = label(mask)  # Label connected components
    regions = regionprops(labeled_mask)

    center_points = []
    for region in regions:
        y_min, x_min, y_max, x_max = region.bbox  # Get bounding box (row, col, row, col)
        if (x_max - x_min) * (y_max - y_min) > 100:  # Filter out small BB
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            center_points.append([x_center, y_center])  # Append center point (x, y)
    
    return center_points

def compute_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    dsc = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if pred_mask.sum() + gt_mask.sum() > 0 else 0
    return iou, dsc

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

# def infer(medsam_model, img_embed, x, y, H, W, device):
#         coords_1024 = np.array([[[
#             x * 1024 / W,
#             y * 1024 / H
#         ]]])
#         coords_torch = torch.tensor(coords_1024, dtype=torch.float32).to(device)
#         labels_torch = torch.tensor([[1]], dtype=torch.long).to(device)
#         print(coords_torch.shape, labels_torch.shape)
#         point_prompt = (coords_torch, labels_torch)

#         sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
#             points = point_prompt,
#             boxes = None,
#             masks = None,
#         )
#         low_res_logits, _ = medsam_model.mask_decoder(
#             image_embeddings=img_embed, # (B, 256, 64, 64)
#             image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
#             sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
#             dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
#             multimask_output=False,
#         )

#         low_res_probs = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
#         low_res_pred = F.interpolate(
#             low_res_probs,
#             size = (H, W),
#             mode = 'bilinear',
#             align_corners = False
#         )
#         low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()

#         seg = np.uint8(low_res_pred > 0.1)

#         return seg

device = "cpu"
medsam_model = sam_model_registry["vit_b"](checkpoint="C:\\Users\\cbaldini\\Desktop\\Segmentation\\MedSAM\\work_dir\\sam_model_laryngoscope_best_converted.pth") #"/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250417-1556/medsam_freezedDEC_model_best_converted.pth") #"/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250401-1801/medsam_model_best_converted.pth") #was "/work/cbaldini/medSAM/code/MedSAM/medsam_vit_b.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()

# Create an automatic mask generator to run SAM without explicit prompts.
# Adjust points_per_side and thresholds to control proposal density and quality.
mask_generator = SamAutomaticMaskGenerator(
    medsam_model,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    stability_score_offset=1,
)

# Paths
img_path = "C:\\Users\\cbaldini\\Desktop\\Segmentation\\AIRCARE-WP4\\New_dataset\\test\\images" #"/work/cbaldini/medSAM/code/Genova set/images" #"/work/cbaldini/medSAM/code/Seoul set/images/all"
mask_path = "C:\\Users\\cbaldini\\Desktop\\Segmentation\\AIRCARE-WP4\\New_dataset\\test\\masks" #"/work/cbaldini/medSAM/code/Seoul set/masks"
seg_path="C:\\Users\\cbaldini\\Desktop\\Segmentation\\AIRCARE-WP4\\segSAMfn_test_newdataset" #"/work/cbaldini/medSAM/code/MedSAM/assets/SEOUL_test_medsam_ENCfreezed"
os.makedirs(seg_path, exist_ok=True)

# metrics_file = os.path.join(seg_path, "metrics.txt")
# with open(metrics_file, "w") as f:
#     f.write("Image, IoU, DSC\n")
import pandas as pd

metrics_file = os.path.join(seg_path, "metrics_testlaryngoscope_msam_fn_all.xlsx")
metrics_list = [] 

for i in os.listdir(img_path):
    # if 'LAR' in i:
    if not 'null' in i:
        img_np = io.imread(join(img_path,i))
        # print(img_np.shape)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif img_np.shape[2]==4:
            img_3c=img_np[:,:,0:3]
        else:
            img_3c = img_np
        print(img_3c.shape)
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

        print(f"Valori immagine normalizzata: min={img_1024.min()}, max={img_1024.max()}")


        mask_np = io.imread(os.path.join(mask_path, i), as_gray=True) > 0
        H, W = img_np.shape[:2]
        box_np = np.array(extract_bounding_box(mask_np))
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        center_points = np.array(extract_center_points(mask_np))
        center_points_1024 = center_points / np.array([W, H]) * 1024

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

        # Visualization with all bounding boxes and separate masks
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        show_mask(mask_np, ax[0], random_color=True)  # Overlay mask

        # Show all bounding boxes on the original image
        for idx, box in enumerate(box_1024):
        # for idx, center in enumerate(center_points):
            medsam_seg = medsam_inference(medsam_model, image_embedding, box[None, :], H, W)
            
            # print(center)
            # print(center[None, 0][0])
            # medsam_seg = infer(medsam_model, image_embedding, center[None, 0][0], center[None, 1][0], H, W, device)

            # medsam_seg_post = keep_largest_connected_component(medsam_seg)  

            # Extract corresponding region from ground truth mask
            x_min, y_min, x_max, y_max = (box * np.array([W, H, W, H]) / 1024).astype(int)
            gt_mask_crop = mask_np[y_min:y_max, x_min:x_max]  # Ground truth for this object
            pred_mask_crop = medsam_seg[y_min:y_max, x_min:x_max]  # MedSAM result for this object

            iou, dsc = compute_metrics(pred_mask_crop, gt_mask_crop)
            
            # x_center, y_center = (center * np.array([W, H]) / 1024).astype(int)
            # gt_mask_crop = mask_np[max(0, y_center-50):y_center+50, max(0, x_center-50):x_center+50]
            # pred_mask_crop = medsam_seg_post[max(0, y_center-50):y_center+50, max(0, x_center-50):x_center+50]

            # # iou, dsc = compute_metrics(pred_mask_crop, gt_mask_crop)
            # iou, dsc = compute_metrics(medsam_seg_post, mask_np) #No crop

            # # Save metrics per object
            # with open(metrics_file, "a") as f:
            #     f.write(f"{i}_obj{idx}, {iou:.4f}, {dsc:.4f}\n")
            metrics_list.append({"Image": i, "Object": idx, "IoU": iou, "DSC": dsc})

            show_mask(medsam_seg, ax[1])  # Overlay mask
            show_box((x_min, y_min, x_max, y_max), ax[1])  # Show bounding box
            # ax[1].scatter(center[None, 0][0], center[None, 1][0], color="blue", s=10)

        ax[1].set_title(f"SAM Segmentation")

        plt.savefig(os.path.join(seg_path, "seg_" + i))

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.title("Immagine Originale")
        # plt.imshow(img_3c)

        # plt.subplot(1, 3, 2)
        # plt.title("Maschera Ground Truth")
        # plt.imshow(mask_np, cmap="gray")

        # plt.subplot(1, 3, 3)
        # plt.title("Maschera Generata")
        # plt.imshow(medsam_seg_post, cmap="gray")
        # plt.savefig(os.path.join(seg_path, "check_" + i))
        
        # for box in box_np:
        #     show_box(box, ax[0])  
        # ax[0].set_title("Input Image and Bounding Boxes")

        # ax[1].imshow(img_3c)

        # # Process each bounding box separately
        # box_original = (box_1024 * np.array([W, H, W, H]) / 1024).astype(int)
        # for box in box_1024:
        #     medsam_seg = medsam_inference(medsam_model, image_embedding, box[None, :], H, W)
        #     show_mask(medsam_seg, ax[1])  # Overlay each mask separately
        # for box in box_original:
        #     show_box(box, ax[1])

        # iou, dsc = compute_metrics(medsam_seg, mask_np)
        # with open(metrics_file, "a") as f:
        #     f.write(f"{i}, {iou:.4f}, {dsc:.4f}\n")

        # ax[1].set_title(f"MedSAM Segmentation after tuning: IoU={iou:.2f}, DSC={dsc:.2f}")

        # plt.savefig(os.path.join(seg_path, "seg_" + i))

df = pd.DataFrame(metrics_list)

# Compute mean IoU and DSC
mean_iou = df["IoU"].mean()
mean_dsc = df["DSC"].mean()

# Append mean values
df = pd.concat([df, pd.DataFrame([{"Image": "Mean", "Object": "-", "IoU": mean_iou, "DSC": mean_dsc}])], ignore_index=True)

# Save to Excel
df.to_excel(metrics_file, index=False)

print(f"Metrics saved to {metrics_file}")
print(f"Mean IoU: {mean_iou:.4f}, Mean DSC: {mean_dsc:.4f}")