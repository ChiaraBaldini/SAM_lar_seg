import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from skimage import io, measure, transform
from segment_anything import sam_model_registry
from ultralytics import YOLO
import pandas as pd
import cv2

yolo_model = YOLO("/work/cbaldini/YOLO/code/HNSCC3d/From old experiments/exp5_lesioNs_GE+CH+IND/weights/best.pt") # "/work/cbaldini/YOLO/code/HNSCC3d/exp8n_coco_GE+CH+IN_mixup_mosaic_copypaste/weights/best.pt"

def extract_bounding_box(mask):
    labeled_mask = measure.label(mask)  # Label connected components
    regions = measure.regionprops(labeled_mask)

    bboxes = []
    for region in regions:
        y_min, x_min, y_max, x_max = region.bbox  # Get bounding box (row, col, row, col)
        if (x_max-x_min)*(y_max-y_min) > 100:   # Filter out small BB
            bboxes.append([x_min, y_min, x_max, y_max])  # Convert to (x_min, y_min, x_max, y_max)
    return bboxes

def compute_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    dsc = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if pred_mask.sum() + gt_mask.sum() > 0 else 0
    return iou, dsc

def show_mask(mask, ax, color, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
        color=color
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

#SAM model
device = "cuda:0"
medsam_model = sam_model_registry["vit_b"](checkpoint="/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250401-1801/medsam_model_best_converted.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()

img_path = "/work/cbaldini/medSAM/code/Genova set/100 LAR selected/images" # "/work/cbaldini/medSAM/code/Genova set/images"
mask_path = "/work/cbaldini/medSAM/code/Genova set/100 LAR selected/GT-A" # "/work/cbaldini/medSAM/code/Genova set/masks"
seg_path = "/work/cbaldini/medSAM/code/MedSAM/assets/GENOVA_test_medSAMyolo100v1_BBperturbed"
os.makedirs(seg_path, exist_ok=True)

metrics_list = [] 
no_det = 0

def get_positive_point(mask):
    props = measure.regionprops(measure.label(mask.astype(int)))
    if not props:
        return None
    y, x = props[0].centroid
    return [int(x), int(y)] 

# def get_non_black_bbox(img, threshold=30):
#     gray = np.mean(img, axis=2)
#     mask = gray > threshold
#     coords = np.argwhere(mask)
#     if coords.size == 0:
#         return None
#     y0, x0 = coords.min(axis=0)
#     y1, x1 = coords.max(axis=0)
#     return [x0, y0, x1, y1]

def get_edge_based_bbox(img, low_thresh=50, high_thresh=150, mask_thresh=30, window_size=500):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > mask_thresh
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
    density_map = np.zeros_like(gray, dtype=np.float32)
    for y in range(0, gray.shape[0], window_size):
        for x in range(0, gray.shape[1], window_size):
            window = edges_masked[y:y+window_size, x:x+window_size]
            density_map[y:y+window_size, x:x+window_size] = np.sum(window) / (window_size * window_size)
    max_density_coords = np.unravel_index(np.argmax(density_map), density_map.shape)
    y0, x0 = max_density_coords
    y1, x1 = y0 + window_size, x0 + window_size
    y1 = min(y1, gray.shape[0])
    x1 = min(x1, gray.shape[1])
    return [x0, y0, x1, y1]

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

def clip_box(box, W, H):
    #Clip BB coordinate to image limits
    x_min, y_min, x_max, y_max = box
    x_min = max(0, min(W, x_min))
    y_min = max(0, min(H, y_min))
    x_max = max(0, min(W, x_max))
    y_max = max(0, min(H, y_max))
    return [x_min, y_min, x_max, y_max]

# BB perturbation parameters
extend_pixels = 20
shrink_pixels = 5

for i in os.listdir(img_path):
    if not 'null' in i:
        img_np = io.imread(os.path.join(img_path, i))
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_3c = img_np[:, :, 0:3]
        else:
            img_3c = img_np
        
        H, W, _ = img_3c.shape
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        ) 
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)

        mask_np = io.imread(os.path.join(mask_path, i), as_gray=True) > 0
        H, W = img_3c.shape[:2]

        results = yolo_model.predict(img_3c, imgsz=640, conf=0.4)[0]
        box_np = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            # if (x2 - x1) * (y2 - y1) > 100:
            if (x2 - x1) * (y2 - y1) > 1:
                box_np.append([x1, y1, x2, y2])
        box_np = np.array(box_np)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)

        
        # #POINT (CENTROID)
        # if box_np.size == 0:
        #     print(f"No YOLO detection in {i} — using centroid fallback.")
            # point = get_positive_point(mask_np)
            # if point is None:
            #     print(f"Skipping {i} — no valid ground truth.")
            #     continue

            # point_1024 = [coord / orig * 1024 for coord, orig in zip(point, [W, H])]
            # point_coords = torch.tensor([[point_1024]], dtype=torch.float, device=device)
            # point_labels = torch.tensor([[1]], dtype=torch.int, device=device)

            # sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            #     points=(point_coords, point_labels),
            #     boxes=None,
            #     masks=None,
            # )
            # low_res_logits, _ = medsam_model.mask_decoder(
            #     image_embeddings=image_embedding,
            #     image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            #     sparse_prompt_embeddings=sparse_embeddings,
            #     dense_prompt_embeddings=dense_embeddings,
            #     multimask_output=False,
            # )
            # low_res_pred = torch.sigmoid(low_res_logits)
            # low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False).detach().cpu().numpy()
            # low_res_pred = low_res_pred.squeeze()
            # medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

            # iou, dsc = compute_metrics(medsam_seg, mask_np)
            # metrics_list.append({"Image": i, "Object": 0, "IoU": iou, "DSC": dsc})

            # show_mask(medsam_seg, ax[1], color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6]))

        # #FULL BOX
        # if box_np.size == 0:
        #     print(f"No YOLO detection in {i} — using full-image fallback.")
        #     box_full = np.array([[0, 0, W, H]])
        #     box_1024 = box_full / np.array([W, H, W, H]) * 1024

        #     medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

        #     iou, dsc = compute_metrics(medsam_seg, mask_np)
        #     metrics_list.append({"Image": i, "Object": 0, "IoU": iou, "DSC": dsc})

        #     show_mask(medsam_seg, ax[1], color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6]))

        #GRID BOX
        if box_np.size == 0:
            print(f"No YOLO detection in {i} — using grid box fallback.")
            no_det+=1
            # box = get_non_black_bbox(img_3c)
            # if box is None:
            #     print(f"Skipping {i} — could not determine non-black area.")
            #     continue
            
            # box_1024 = np.array(box) / np.array([W, H, W, H]) * 1024

            # medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024[None, :], H, W)
            # iou, dsc = compute_metrics(medsam_seg, mask_np)
            # metrics_list.append({"Image": i, "Object": 0, "IoU": iou, "DSC": dsc})
            # box = get_edge_based_bbox(img_3c)
            # if box is None:
            #     print(f"Skipping {i} — no edges found.")
            #     continue

            # box_1024 = np.array(box) / np.array([W, H, W, H]) * 1024
            # medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024[None, :], H, W)

            # iou, dsc = compute_metrics(medsam_seg, mask_np)
            # metrics_list.append({"Image": i, "Object": 0, "IoU": iou, "DSC": dsc})

            # show_box(box, ax[1])
            # show_mask(medsam_seg, ax[1], color = np.array([133 / 255, 120 / 255, 50 / 255, 0.6]))


        else:
            # # ONLY ORIGINAL BB FROM YOLO
            # for idx, box in enumerate(box_np):
            #     box_1024 = box / np.array([W, H, W, H]) * 1024 
            #     medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024[None, :], H, W)

            #     # final_mask = np.maximum(final_mask, medsam_seg)

            #     iou, dsc = compute_metrics(medsam_seg, mask_np)
            #     metrics_list.append({"Image": i, "Object": idx, "IoU": iou, "DSC": dsc})

            #     show_mask(medsam_seg, ax[1], color = np.array([251 / 255, 52 / 255, 30 / 255, 0.6]))
            #     show_box(box, ax[1])
            #     # ax[1].set_title(f"MedSAM Segmentation (Object {idx})")
            #     # plt.savefig(os.path.join(seg_path, "seg_" + i))
            # ax[0].set_title("Original Image")
            # ax[1].set_title("YOLO + MedSAM segmentation")
            # plt.savefig(os.path.join(seg_path, "seg_all_" + i))

            # BB PERTURBATION
            for idx, box in enumerate(box_np):
                # Original bounding box
                box_1024_original = box / np.array([W, H, W, H])

                
                # Extended bounding box (e.g., +30 pixels in each direction)
                box_extended = clip_box([
                    box[0] - extend_pixels,  # x_min - 30
                    box[1] - extend_pixels,  # y_min - 30
                    box[2] + extend_pixels,  # x_max + 30
                    box[3] + extend_pixels   # y_max + 30
                ], W, H)
                box_1024_extended = np.array(box_extended) / np.array([W, H, W, H]) * 1024

                # Shrunken bounding box (e.g., -10 pixels in each direction)
                box_shrunken = clip_box([
                    min(W, max(0, box[0] + shrink_pixels)),  # x_min + 10
                    min(H, max(0, box[1] + shrink_pixels)),  # y_min + 10
                    max(0, min(W, box[2] - shrink_pixels)),  # x_max - 10
                    max(0, min(H, box[3] - shrink_pixels)),  # y_max - 10
                ], W, H)
                box_1024_shrunken = np.array(box_shrunken) / np.array([W, H, W, H]) * 1024

                # Run inference for each bounding box
                mask_original = medsam_inference(medsam_model, image_embedding, box_1024_original[None, :], H, W)
                mask_extended = medsam_inference(medsam_model, image_embedding, box_1024_extended[None, :], H, W)
                mask_shrunken = medsam_inference(medsam_model, image_embedding, box_1024_shrunken[None, :], H, W)

                # Combine the masks (logical OR or average and threshold)
                # combined_mask = np.maximum.reduce([mask_original, mask_extended, mask_shrunken])
                combined_mask = (mask_original + mask_extended + mask_shrunken) / 3
                combined_mask = (combined_mask > 0.5).astype(np.uint8)

                # Compute metrics for the combined mask
                iou, dsc = compute_metrics(combined_mask, mask_np)
                metrics_list.append({"Image": i, "Object": idx, "IoU": iou, "DSC": dsc})

                # Visualize the combined mask
                show_mask(combined_mask, ax[1], color=np.array([251 / 255, 52 / 255, 30 / 255, 0.6]))
                show_box(box, ax[1])

            ax[0].set_title("Original Image")
            ax[1].set_title("YOLO + MedSAM Combined Segmentation")
            plt.savefig(os.path.join(seg_path, "seg_all_BBperurbed" + i))

        # ax[0].set_title("Original Image")
        # ax[1].set_title("YOLO + MedSAM segmentation")
        # plt.savefig(os.path.join(seg_path, "seg_all_" + i))

df = pd.DataFrame(metrics_list)
mean_iou = df["IoU"].mean()
mean_dsc = df["DSC"].mean()

df = pd.concat([df, pd.DataFrame([{"Image": "Mean", "Object": "-", "IoU": mean_iou, "DSC": mean_dsc}])], ignore_index=True)
df = pd.concat([df, pd.DataFrame([{"Image": "NO_DETCETION", "Object": no_det, "IoU": "-", "DSC": "-"}])], ignore_index=True)
df.to_excel(os.path.join(seg_path, "metrics_testGE_medSAMyolo100v1_BBperturbed.xlsx"), index=False)

print(f"Metrics saved to metrics_testGE_medSAMyolo100v1_BBperturbed.xlsx")
print(f"Mean IoU: {mean_iou:.4f}, Mean DSC: {mean_dsc:.4f}")