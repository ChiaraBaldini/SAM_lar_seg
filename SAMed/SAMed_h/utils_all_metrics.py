import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_iou(pred, gt):
    """
    Calculate Intersection over Union (IoU) for binary segmentation
    
    Args:
        pred: predicted mask (binary)
        gt: ground truth mask (binary)
    
    Returns:
        float: IoU score
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        # If both masks are empty, IoU = 1
        return 1.0 if intersection == 0 else 0.0
    else:
        return intersection / union


def calculate_metric_percase(pred, gt):
    """
    Calculate metrics per case including DSC, HD95, and IoU
    
    Args:
        pred: predicted mask
        gt: ground truth mask
    
    Returns:
        tuple: (dice, hd95, iou)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    # Calculate IoU
    iou = calculate_iou(pred, gt)
    
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95, iou
    elif pred.sum() > 0 and gt.sum() == 0:
        # False positive case
        return 0, 0, 0  # No true positives, so DSC=0, HD95=0, IoU=0
    else:
        # True negative case (both empty) or False negative case
        if pred.sum() == 0 and gt.sum() == 0:
            return 1, 0, 1  # Both empty: DSC=1, HD95=0, IoU=1
        else:
            return 0, 0, 0  # False negative: DSC=0, HD95=0, IoU=0


def calculate_additional_metrics(pred, gt):
    """
    Calculate additional metrics: Sensitivity, Specificity, Precision
    
    Args:
        pred: predicted mask (binary)
        gt: ground truth mask (binary)
    
    Returns:
        dict: dictionary with additional metrics
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # Calculate confusion matrix components
    tp = np.logical_and(pred, gt).sum()          # True Positives
    fp = np.logical_and(pred, ~gt).sum()         # False Positives
    fn = np.logical_and(~pred, gt).sum()         # False Negatives
    tn = np.logical_and(~pred, ~gt).sum()        # True Negatives
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Precision
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    
    # Save label visualization
    if label.max() > 0:
        label1 = (label - label.min()) / (label.max() - label.min()) * 255
    else:
        label1 = label.copy()
    label1 = label1.astype(np.uint8)
    
    if test_save_path is not None:
        if not os.path.exists(os.path.join(test_save_path,'images/label')):
            os.makedirs(os.path.join(test_save_path,'images/label'))
        imageio.imwrite(os.path.join(test_save_path,f'images/label/label_{case}.png'), label1)
    
    # Process 3D volume (slice by slice)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[2]):
            slice = image[:, :, ind]
            x, y = slice.shape[0], slice.shape[1]
            
            # Resize to input size
            if x != input_size[0] or y != input_size[1]:
                slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)
            
            new_x, new_y = slice.shape[0], slice.shape[1]
            
            # Resize to patch size
            if new_x != patch_size[0] or new_y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)
            
            # Prepare input tensor
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            print(f"Input tensor shape: {inputs.shape}")
            
            # Model inference
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                print(f"Output masks shape: {output_masks.shape}")
                
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                print(f"Argmax output shape: {out.shape}")
                
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                
                # Resize back to original dimensions
                if x != out_h or y != out_w:
                    pred = zoom(out, (x/out_h, y/out_w), order=0)
                else:
                    pred = out
                    
                print(f"Final prediction shape: {pred.shape}")
                prediction = pred
        
        # Save prediction visualization
        if test_save_path is not None:
            prediction1 = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255
            prediction1 = prediction1.astype(np.uint8)
            if not os.path.exists(os.path.join(test_save_path,'images/pred')):
                os.makedirs(os.path.join(test_save_path,'images/pred'))
            assert prediction.shape[0] == label.shape[0]
            imageio.imwrite(os.path.join(test_save_path,f'images/pred/pred_{case}.png'), prediction1)
    
    # Process 2D image
    else:
        x, y = image.shape[-2:]
        
        # Resize to patch size
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        
        # Prepare input tensor
        inputs = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        
        # Model inference
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            
            # Resize back to original dimensions
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    
    # Calculate metrics for each class
    metric_list = []
    additional_metrics_list = []
    
    for i in range(1, classes + 1):
        pred_class = (prediction == i).astype(np.uint8)
        gt_class = (label == i).astype(np.uint8)
        
        # # Calculate main metrics (DSC, HD95, IoU)
        # dice, hd95, iou = calculate_metric_percase(pred_class.copy(), gt_class.copy())
        # metric_list.append([dice, hd95, iou])

        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        
        # Calculate additional metrics
        # additional_metrics = calculate_additional_metrics(pred_class, gt_class)
        # additional_metrics_list.append(additional_metrics)
        
        # print(f"Class {i} - DSC: {dice:.4f}, HD95: {hd95:.4f}, IoU: {iou:.4f}")
        # print(f"Class {i} - Sensitivity: {additional_metrics['sensitivity']:.4f}, "
        #       f"Specificity: {additional_metrics['specificity']:.4f}, "
        #       f"Precision: {additional_metrics['precision']:.4f}")

    # Save results to files
    if test_save_path is not None:
        # Save 3D volumes
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        
        # Uncomment if you want to save .nii.gz files
        # sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        # sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        # sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        
        # # Save detailed metrics to text file
        # metrics_file = os.path.join(test_save_path, f'metrics_{case}.txt')
        # with open(metrics_file, 'w') as f:
        #     f.write(f"Detailed Metrics for Case: {case}\n")
        #     f.write("=" * 50 + "\n")
        #     for i, (metrics, additional) in enumerate(zip(metric_list, additional_metrics_list)):
        #         f.write(f"\nClass {i+1}:\n")
        #         f.write(f"  DSC: {metrics[0]:.6f}\n")
        #         f.write(f"  HD95: {metrics[1]:.6f}\n")
        #         f.write(f"  IoU: {metrics[2]:.6f}\n")
        #         f.write(f"  Sensitivity: {additional['sensitivity']:.6f}\n")
        #         f.write(f"  Specificity: {additional['specificity']:.6f}\n")
        #         f.write(f"  Precision: {additional['precision']:.6f}\n")
        #         f.write(f"  TP: {additional['tp']}\n")
        #         f.write(f"  FP: {additional['fp']}\n")
        #         f.write(f"  FN: {additional['fn']}\n")
        #         f.write(f"  TN: {additional['tn']}\n")
    
    return metric_list # , additional_metrics_list


def summarize_metrics(all_metrics, all_additional_metrics, class_names=None):
    """
    Summarize metrics across all test cases
    
    Args:
        all_metrics: list of metric lists from all cases
        all_additional_metrics: list of additional metrics from all cases
        class_names: dict mapping class indices to names
    
    Returns:
        dict: summary statistics
    """
    if not all_metrics:
        return {}
    
    num_classes = len(all_metrics[0])
    summary = {}
    
    for class_idx in range(num_classes):
        class_name = class_names.get(class_idx + 1, f"Class_{class_idx + 1}") if class_names else f"Class_{class_idx + 1}"
        
        # Extract metrics for this class across all cases
        dice_scores = [metrics[class_idx][0] for metrics in all_metrics]
        hd95_scores = [metrics[class_idx][1] for metrics in all_metrics]
        iou_scores = [metrics[class_idx][2] for metrics in all_metrics]
        
        sensitivity_scores = [add_metrics[class_idx]['sensitivity'] for add_metrics in all_additional_metrics]
        specificity_scores = [add_metrics[class_idx]['specificity'] for add_metrics in all_additional_metrics]
        precision_scores = [add_metrics[class_idx]['precision'] for add_metrics in all_additional_metrics]
        
        summary[class_name] = {
            'DSC': {
                'mean': np.mean(dice_scores),
                'std': np.std(dice_scores),
                'min': np.min(dice_scores),
                'max': np.max(dice_scores)
            },
            'HD95': {
                'mean': np.mean(hd95_scores),
                'std': np.std(hd95_scores),
                'min': np.min(hd95_scores),
                'max': np.max(hd95_scores)
            },
            'IoU': {
                'mean': np.mean(iou_scores),
                'std': np.std(iou_scores),
                'min': np.min(iou_scores),
                'max': np.max(iou_scores)
            },
            'Sensitivity': {
                'mean': np.mean(sensitivity_scores),
                'std': np.std(sensitivity_scores),
                'min': np.min(sensitivity_scores),
                'max': np.max(sensitivity_scores)
            },
            'Specificity': {
                'mean': np.mean(specificity_scores),
                'std': np.std(specificity_scores),
                'min': np.min(specificity_scores),
                'max': np.max(specificity_scores)
            },
            'Precision': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores),
                'min': np.min(precision_scores),
                'max': np.max(precision_scores)
            }
        }
    
    return summary