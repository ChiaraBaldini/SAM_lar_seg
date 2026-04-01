import torch
from torch import nn
import numpy as np
import os
import glob
import torchvision
import matplotlib.pyplot as plt
import random
from segment_anything import sam_model_registry

# Dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, image_folder, mask_folder, set_type="train"):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.base_path = base_path
        self.device = device
        self.resize = torchvision.transforms.Resize(
            (1024, 1024),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        self.all_img_files = sorted(glob.glob(os.path.join(self.base_path, self.image_folder, "**/*.png"), recursive=True))
        # Split dataset: 90% train, 10% test (puoi cambiare la percentuale)
        split_idx = int(len(self.all_img_files) * 0.9)
        if set_type == "train":
            self.img_files = self.all_img_files[:split_idx]
        else:
            self.img_files = self.all_img_files[split_idx:]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_path = self.img_files[index]
        # Assumiamo che il nome della maschera sia identico a quello dell'immagine
        mask_name = os.path.basename(image_path)
        mask_path = os.path.join(self.base_path, self.mask_folder, mask_name)
        image = torchvision.io.read_image(image_path)
        mask = torchvision.io.read_image(mask_path)
        image = self.resize(image)
        mask = self.resize(mask)
            # Forza l'immagine a 3 canali
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]  # Scarta il canale alpha
        elif image.shape[0] != 3:
            raise ValueError(f"Image {image_path} has {image.shape[0]} channels (expected 3 or 1 or 4)")
        # Forza la maschera a 1 canale
        if mask.shape[0] > 1:
            mask = mask[0:1, :, :]
        mask = mask.type(torch.float) / 255
        image = image.type(torch.float)
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Image: {image_path}, Mask: {mask_path}")
        return image, mask
        

# LayerNorm2d come da repo SAM
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# Decoder custom
class SAM_Decoder(nn.Module):
    def __init__(self, sam_encoder, sam_preprocess):
        super().__init__()
        self.sam_encoder = sam_encoder
        self.sam_preprocess = sam_preprocess
        for layer_no, param in enumerate(self.sam_encoder.parameters()):
            if(layer_no > (last_layer_no - 6)):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.nn_drop = nn.Dropout(p=0.2)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.norm3 = LayerNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.norm4 = LayerNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.sam_preprocess(x)
        x = self.sam_encoder(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.nn_drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = torch.nn.functional.relu(x)
        x = self.nn_drop(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    # Impostazione seed per la riproducibilità
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Costanti
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = r"C:\Users\cbaldini\Desktop\Segmentation"
    save_path = r"C:\Users\cbaldini\Desktop\Segmentation\MedSAM\custom_decoder"
    images_folder = r"100 LAR selected\images"
    masks_folder = r"100 LAR selected\GT-A"
    batch_size = 8
    epochs = 11
    t2_batch_size = 1
    train_split = 0.1

    # Caricamento modello SAM
    sam = sam_model_registry["vit_b"](checkpoint=r"C:\Users\cbaldini\Desktop\Segmentation\MedSAM\medsam_vit_b.pth")
    sam = sam.to(device)

    

    # Trova l'ultimo layer della neck Conv2d
    for layer_no, param in enumerate(sam.image_encoder.parameters()):
        pass
    last_layer_no = layer_no


    sam_decoder = SAM_Decoder(sam_encoder=sam.image_encoder, sam_preprocess=sam.preprocess)
    sam_decoder = sam_decoder.to(device)

    
    # Split dataset
    dataset = ImageDataset(base_path, images_folder, masks_folder, set_type="train")
    train_dataset, t1_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*train_split), len(dataset)-int(len(dataset)*train_split)])
    t2_dataset = ImageDataset(base_path, images_folder, masks_folder, set_type="test")

    # DataLoader
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, prefetch_factor=3)
    t1_data_loader = torch.utils.data.DataLoader(t1_dataset, batch_size=batch_size, shuffle=True, num_workers=5, prefetch_factor=3)
    t2_data_loader = torch.utils.data.DataLoader(t2_dataset, batch_size=t2_batch_size, shuffle=True, num_workers=5, prefetch_factor=3)

    # Loss e ottimizzatore
    bce_loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(sam_decoder.parameters(), lr=0.01)

    # Training loop
    total_steps = len(train_dataset) // batch_size
    mini_batch_event = int(total_steps * 0.25)
    for epoch in range(epochs):
        sam_decoder.train()
        epoch_loss = 0
        mini_event_loss = 0
        for i, data in enumerate(train_data_loader, 0):
            images, masks = data
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            pred_masks = sam_decoder(images)
            loss = bce_loss(pred_masks, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            mini_event_loss += loss.item()
            if i % mini_batch_event == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {mini_event_loss / (mini_batch_event+1):.3f}')
                mini_event_loss = 0
        print(f'=====> [{epoch + 1}, {i + 1:5d}] loss: {epoch_loss / (total_steps+1):.3f}')

    # Inference esempio
    sam_decoder.eval()
    with torch.no_grad():
        inpt_0, gt_0 = t2_dataset[0][0], t2_dataset[0][1]
        decoder_opt = sam_decoder(inpt_0.to(device).unsqueeze(0))
        decoder_opt_np = ((decoder_opt > 0.5)*1).to("cpu").numpy()[0].transpose(1,2,0)
        gt_0_np = gt_0.to("cpu").numpy().transpose(1,2,0)
        temp_img_np = inpt_0.to("cpu").numpy()
        temp_img_np = np.transpose(temp_img_np, [1,2,0]).astype(np.uint8)
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(gt_0_np)
        axs[0].axis('off')
        axs[1].imshow(decoder_opt_np)
        axs[1].axis('off')
        axs[2].imshow(temp_img_np)
        axs[2].axis('off')
        fig.tight_layout()
        plt.show()

    # Calcolo IOU su t2
    iou_loss_li = []
    with torch.no_grad():
        t2_loss = 0
        for i, test_data in enumerate(t2_data_loader, 0):
            test_inputs, test_labels = test_data
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            pred_masks = sam_decoder(test_inputs)
            t2_step_loss = bce_loss(pred_masks, test_labels)
            t2_loss += t2_step_loss
            intersection = torch.logical_and((pred_masks > 0.5)*1.0, test_labels)
            union = torch.logical_or((pred_masks > 0.5)*1.0, test_labels)
            iou = torch.sum(intersection) / torch.sum(union)
            iou_loss_li.append(iou.item())
        print(f'-------------> Test T2 Loss: {t2_loss / len(t2_data_loader):.3f}')
        print("IOU LOSS: ", sum(iou_loss_li)/(len(iou_loss_li)))

    # Salvataggio modello
    torch.save(sam_decoder.state_dict(), os.path.join(save_path, "sam_enc_custom_decoder.pt"))