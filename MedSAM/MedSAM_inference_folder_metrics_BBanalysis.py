import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from skimage import io, measure, transform
from segment_anything import sam_model_registry
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

join = os.path.join

import argparse
from skimage.measure import label, regionprops
import numpy as np
from skimage.measure import label

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import cv2
import time

def extract_bounding_box(mask):
    """Extract bounding boxes from mask and return with area information"""
    labeled_mask = label(mask)  
    regions = regionprops(labeled_mask)

    bboxes = []
    for region in regions:
        y_min, x_min, y_max, x_max = region.bbox  
        area = (x_max - x_min) * (y_max - y_min)
        if area > 100:   # Filter out small bounding boxes
            bboxes.append([x_min, y_min, x_max, y_max, area])  # Added area as 5th element
    
    return bboxes

def categorize_bbox_size(area, percentiles):
    """Categorize bounding box based on area percentiles"""
    if area <= percentiles[0]:
        return "Small"
    elif area <= percentiles[1]:
        return "Medium"
    else:
        return "Large"

def compute_metrics(pred_mask, gt_mask):
    """Compute IoU and DSC metrics"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    dsc = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if pred_mask.sum() + gt_mask.sum() > 0 else 0
    return iou, dsc

def show_mask(mask, ax, color, random_color=False):
    """Display mask overlay on image"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = color
        # color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, color="blue", linewidth=2):
    """Display bounding box on image"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=linewidth)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    """Perform MedSAM inference with bounding box prompt"""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    # # Extract bb center
    # x_min, y_min, x_max, y_max = box_torch[0, 0].cpu().numpy()
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # coords = torch.tensor([[[x_center, y_center]]], device=img_embed.device)  # (1, 1, 2)
    # labels = torch.tensor([[1]], device=img_embed.device)  # (1, 1) - Example label (1 for foreground)
    # points = (coords, labels)
    # sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
    #     points=points,
    #     boxes=None,
    #     masks=None,
    # )
    
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

def plot_performance_vs_bbox_size(df, seg_path):
    """Create plots showing performance vs bounding box size"""
    
    # Filter out mean row and ensure numeric data
    df_filtered = df[df["Image"] != "Mean"].copy()
    df_filtered["IoU"] = pd.to_numeric(df_filtered["IoU"])
    df_filtered["DSC"] = pd.to_numeric(df_filtered["DSC"])
    df_filtered["BB_Area"] = pd.to_numeric(df_filtered["BB_Area"])
    df_filtered["BB_Width"] = pd.to_numeric(df_filtered["BB_Width"])
    df_filtered["BB_Height"] = pd.to_numeric(df_filtered["BB_Height"])
    
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: IoU vs BB Area (scatter plot)
    axes[0, 0].scatter(df_filtered["BB_Area"], df_filtered["IoU"], alpha=0.6, color='blue')
    axes[0, 0].set_xlabel("Bounding Box Area (pixels²)")
    axes[0, 0].set_ylabel("IoU")
    axes[0, 0].set_title("IoU vs Bounding Box Area")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_filtered["BB_Area"], df_filtered["IoU"], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(df_filtered["BB_Area"], p(df_filtered["BB_Area"]), "r--", alpha=0.8)
    
    # Plot 2: DSC vs BB Area (scatter plot)
    axes[0, 1].scatter(df_filtered["BB_Area"], df_filtered["DSC"], alpha=0.6, color='green')
    axes[0, 1].set_xlabel("Bounding Box Area (pixels²)")
    axes[0, 1].set_ylabel("DSC")
    axes[0, 1].set_title("DSC vs Bounding Box Area")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df_filtered["BB_Area"], df_filtered["DSC"], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df_filtered["BB_Area"], p(df_filtered["BB_Area"]), "r--", alpha=0.8)
    
    # Plot 3: Performance by BB Size Category (box plot)
    metrics_melted = df_filtered.melt(
        id_vars=["BB_Size_Category"], 
        value_vars=["IoU", "DSC"], 
        var_name="Metric", 
        value_name="Score"
    )
    
    sns.boxplot(data=metrics_melted, x="BB_Size_Category", y="Score", 
                hue="Metric", ax=axes[0, 2])
    axes[0, 2].set_title("Performance by Bounding Box Size Category")
    axes[0, 2].set_xlabel("Bounding Box Size Category")
    axes[0, 2].set_ylabel("Score")
    
    # Plot 4: IoU vs BB Width
    axes[1, 0].scatter(df_filtered["BB_Width"], df_filtered["IoU"], alpha=0.6, color='purple')
    axes[1, 0].set_xlabel("Bounding Box Width (pixels)")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].set_title("IoU vs Bounding Box Width")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: IoU vs BB Height
    axes[1, 1].scatter(df_filtered["BB_Height"], df_filtered["IoU"], alpha=0.6, color='orange')
    axes[1, 1].set_xlabel("Bounding Box Height (pixels)")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("IoU vs Bounding Box Height")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Aspect Ratio vs Performance
    df_filtered["Aspect_Ratio"] = df_filtered["BB_Width"] / df_filtered["BB_Height"]
    axes[1, 2].scatter(df_filtered["Aspect_Ratio"], df_filtered["IoU"], alpha=0.6, color='red')
    axes[1, 2].set_xlabel("Aspect Ratio (Width/Height)")
    axes[1, 2].set_ylabel("IoU")
    axes[1, 2].set_title("IoU vs Bounding Box Aspect Ratio")
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(seg_path, "performance_vs_bbox_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    summary_stats = df_filtered.groupby("BB_Size_Category").agg({
        "IoU": ["count", "mean", "std", "min", "max"],
        "DSC": ["count", "mean", "std", "min", "max"],
        "BB_Area": ["mean", "min", "max"]
    }).round(4)
    
    # Save summary statistics
    summary_stats.to_excel(os.path.join(seg_path, "bbox_size_analysis_summary.xlsx"))
    
    print("\n=== BOUNDING BOX SIZE ANALYSIS ===")
    print(summary_stats)
    
    return summary_stats

def show_contour_mask(mask, ax, color, random_color=False, thickness=2):
    """Display only the contour of the mask on the image"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        color = color

    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image for the contours
    h, w = mask.shape[-2:]
    contour_image = np.zeros((h, w, 4), dtype=np.float32)

    # Draw contours on the empty image
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, color, thickness=thickness)

    # Overlay the contour image on the axis
    ax.imshow(contour_image)

# Load model and setup
device = "cuda:0"
medsam_model = sam_model_registry["vit_b"](checkpoint="/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") # /work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") #/work/cbaldini/medSAM/code/sam_vit_b_01ec64.pth") # /work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") # "/work/cbaldini/medSAM/code/MedSAM/medsam_vit_b.pth") #/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") # /work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") # /work/cbaldini/medSAM/code/MedSAM/medsam_vit_b.pth") #/work/cbaldini/medSAM/code/MedSAM/work_dir/MedSAM-ViT-B-20250611-0720/medsam_model_GE_best_converted.pth") /work/cbaldini/medSAM/code/MedSAM/work_dir/SAM-ViT-B-20250610-1002/sam_model_GE_best_converted.pth") #
medsam_model = medsam_model.to(device)
medsam_model.eval()

# Paths
img_path = "/work/cbaldini/medSAM/code/Seoul set/all/images/" #/work/cbaldini/medSAM/code/Genova set/henance/easy_margins/images"
mask_path = "/work/cbaldini/medSAM/code/Seoul set/all/masks/" #/work/cbaldini/medSAM/code/Genova set/henance/easy_margins/masks"
seg_path = "/work/cbaldini/medSAM/code/MedSAM/assets/SE_test_MedSAM_fn_contour"
os.makedirs(seg_path, exist_ok=True)

metrics_file = os.path.join(seg_path, "metrics_testSE_MedSAM_fn_contour.xlsx")
metrics_list = []

time_file = os.path.join(seg_path, "inference_times.txt")
# Open the file to write inference times
with open(time_file, "w") as f:
    f.write("Image,Inference Time (seconds)\n")

# First pass: collect all bounding box areas to calculate percentiles
all_areas = []
for i in os.listdir(img_path):
    if not 'null' in i:
        mask_np = io.imread(os.path.join(mask_path, i), as_gray=True) > 0
        bboxes_with_area = extract_bounding_box(mask_np)
        for bbox in bboxes_with_area:
            all_areas.append(bbox[4])  # Area is the 5th element

# Calculate percentiles for size categorization
area_percentiles = np.percentile(all_areas, [33, 66]) if all_areas else [0, 0]
print(f"Area percentiles - 33rd: {area_percentiles[0]:.0f}, 66th: {area_percentiles[1]:.0f}")

# Main processing loop
for i in os.listdir(img_path):  # Limit to first 15 images for testing
    if not 'null' in i:
        img_np = io.imread(join(img_path, i))
        
        # Handle different image formats
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_3c = img_np[:, :, 0:3]
        else:
            img_3c = img_np
            
        print(f"Processing {i}: {img_3c.shape}")
        H, W, _ = img_3c.shape

        # Prepare image for model
        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        print(f"Normalized image values: min={img_1024.min()}, max={img_1024.max()}")

        # Load ground truth mask and extract bounding boxes
        mask_np = io.imread(os.path.join(mask_path, i), as_gray=True) > 0
        bboxes_with_area = extract_bounding_box(mask_np)
        
        # Convert bounding boxes to 1024x1024 coordinate system
        boxes_1024 = []
        for bbox in bboxes_with_area:
            box_coords = np.array(bbox[:4]) / np.array([W, H, W, H]) * 1024
            boxes_1024.append(box_coords)


        # Visualization setup
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        # show_mask(mask_np, ax[0], color = np.array([144 / 255, 238 / 255, 144 / 255, 0.6]))
        show_contour_mask(mask_np.astype(np.uint8)*255, ax[0], color=np.array([0 / 255, 0 / 255, 255 / 255, 0.9]), thickness=10)

        final_pred_mask = np.zeros((H, W), dtype=np.uint8)

        # Get image embeddings
        start_time = time.time()
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)
        end_time_img = time.time()
        embedding_time = end_time_img - start_time

        # Process each bounding box
        for idx, (bbox_with_area, box_1024) in enumerate(zip(bboxes_with_area, boxes_1024)):
            # Extract bounding box information
            x_min, y_min, x_max, y_max, area = bbox_with_area
            width = x_max - x_min
            height = y_max - y_min
            size_category = categorize_bbox_size(area, area_percentiles)
            
            # Perform inference
            start_time2 = time.time()
            medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024[None, :], H, W)
            end_time = time.time()
            inference_time = end_time - start_time2
            Total_inference_time = embedding_time + inference_time
            
            with open(time_file, "a") as f:
                f.write(f"{i},{Total_inference_time:.4f}\n")
            
            final_pred_mask = np.logical_or(final_pred_mask, medsam_seg).astype(np.uint8)
            
            # Extract corresponding regions for metric calculation
            gt_mask_crop = mask_np[y_min:y_max, x_min:x_max]
            pred_mask_crop = medsam_seg[y_min:y_max, x_min:x_max]

            # # Compute metrics
            # # iou, dsc = compute_metrics(pred_mask_crop, gt_mask_crop)
            # iou, dsc = compute_metrics(medsam_seg, mask_np)
            
            # # Store comprehensive metrics
            # metrics_list.append({
            #     "Image": i,
            #     "Object": idx,
            #     "IoU": iou,
            #     "DSC": dsc,
            #     "BB_Area": area,
            #     "BB_Width": width,
            #     "BB_Height": height,
            #     "BB_Size_Category": size_category,
            #     "BB_X_Min": x_min,
            #     "BB_Y_Min": y_min,
            #     "BB_X_Max": x_max,
            #     "BB_Y_Max": y_max
            # })

            # Visualization
            # show_mask(medsam_seg, ax[1])
            
            # Color code bounding boxes by size
            bbox_colors = {"Small": "green", "Medium": "orange", "Large": "red"}
            show_box((x_min, y_min, x_max, y_max), ax[1], 
                    color="blue", linewidth=2)

        # show_mask(final_pred_mask, ax[1], color = np.array([255 / 255, 102 / 255, 102 / 255, 0.6]))
        show_contour_mask(final_pred_mask*255, ax[1], color=np.array([255 / 255, 0 / 255, 0 / 255, 0.9]), thickness=10)
        
        # Compute metrics
        mask_save_dir = os.path.join(seg_path, "saved_masks")
        os.makedirs(mask_save_dir, exist_ok=True)
        # Ground truth
        gt_save_path = os.path.join(mask_save_dir, f"{i}_gt_mask.png")
        # cv2.imwrite(gt_save_path, (mask_np * 255).astype(np.uint8))
        # Predicted
        pred_save_path = os.path.join(mask_save_dir, f"{i}_pred_mask.png")
        # cv2.imwrite(pred_save_path, (final_pred_mask * 255).astype(np.uint8))
        # Metrics computation
        iou, dsc = compute_metrics(final_pred_mask, mask_np)
        
        # Store comprehensive metrics
        metrics_list.append({
            "Image": i,
            "IoU": iou,
            "DSC": dsc,
        })

        ax[0].set_title("Original Image with Ground Truth")
        ax[1].set_title("SAM Segmentation Results")
        
        # # Add legend for bounding box colors
        # from matplotlib.patches import Patch
        # legend_elements = [Patch(facecolor='green', label='Small BB'),
        #                   Patch(facecolor='orange', label='Medium BB'),
        #                   Patch(facecolor='red', label='Large BB')]
        # ax[1].legend(handles=legend_elements, loc='upper right')

        plt.savefig(os.path.join(seg_path, f"seg_{i}"), dpi=300, bbox_inches='tight')
        plt.close(fig)

# Create DataFrame and compute statistics
df = pd.DataFrame(metrics_list)

# Compute overall mean IoU and DSC
mean_iou = df["IoU"].mean()
mean_dsc = df["DSC"].mean()

# Append mean values
mean_row = {
    "Image": "Mean", 
    "Object": "-", 
    "IoU": mean_iou, 
    "DSC": mean_dsc,
}
#     "BB_Area": df["BB_Area"].mean(),
#     "BB_Width": df["BB_Width"].mean(),
#     "BB_Height": df["BB_Height"].mean(),
#     "BB_Size_Category": "-",
#     "BB_X_Min": "-",
#     "BB_Y_Min": "-", 
#     "BB_X_Max": "-",
#     "BB_Y_Max": "-"
# }

df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

# Save detailed metrics to Excel
df.to_excel(metrics_file, index=False)

# # Generate performance vs bounding box size analysis
# summary_stats = plot_performance_vs_bbox_size(df, seg_path)

print(f"\nMetrics saved to {metrics_file}")
print(f"Overall Mean IoU: {mean_iou:.4f}, Mean DSC: {mean_dsc:.4f}")
print(f"Performance analysis plots saved to {seg_path}")

# # Print summary by bounding box size category
# print("\n=== PERFORMANCE BY BOUNDING BOX SIZE ===")
# for category in ["Small", "Medium", "Large"]:
#     category_data = df[df["BB_Size_Category"] == category]
#     if len(category_data) > 0:
#         print(f"{category} BBs (n={len(category_data)}):")
#         print(f"  Mean IoU: {category_data['IoU'].mean():.4f} ± {category_data['IoU'].std():.4f}")
#         print(f"  Mean DSC: {category_data['DSC'].mean():.4f} ± {category_data['DSC'].std():.4f}")
#         print(f"  Mean Area: {category_data['BB_Area'].mean():.0f} pixels²")