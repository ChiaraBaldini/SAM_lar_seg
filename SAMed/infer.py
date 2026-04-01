import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from PIL import Image
from icecream import ic


# class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: 'liver', 6: 'stomach', 7: 'aorta', 8: 'pancreas'}
class_to_name={1: 'SCC'}

def inference_predict_only(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='train')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['name'][0]
        pred_mask = test_single_volume(
            image, 
            label=label,
            net=model, 
            classes=args.num_classes, 
            multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size], 
            input_size=[args.input_size, args.input_size],
            test_save_path=test_save_path, 
            case=case_name, 
            z_spacing=db_config['z_spacing']
        )
        # print(f"Predicted mask shape: {pred_mask.shape}")
        # if test_save_path is not None:
        #     png_path = os.path.join(test_save_path, f"{case_name}_pred.png")
        #     img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        #     img.save(png_path)
        
    logging.info("Inference Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='/work/cbaldini/medSAM/code/MedSAM/data/npy/Genova')
    parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--list_dir', type=str, default='/work/cbaldini/medSAM/code/SAMed/lists/lists_Genova', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/work/cbaldini/medSAM/code/SAMed/output/sam/results_epoch99', help='output dir')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/work/cbaldini/medSAM/code/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='/work/cbaldini/medSAM/code/SAMed/output/sam/results/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log_epoch99')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions_epoch99_GE')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference_predict_only(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
