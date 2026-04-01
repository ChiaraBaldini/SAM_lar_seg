import os
import json

def generate_image_to_mask_mapping(images_dir, masks_dir, output_file):
    # Dizionario per mappare immagini e maschere
    image_to_masks = {}

    # Lista di immagini nella cartella
    images = sorted(os.listdir(images_dir))

    # Lista di maschere nella cartella
    masks = sorted(os.listdir(masks_dir))

    # Creazione della mappatura
    for image in images:
        image_name = os.path.splitext(image)[0]  # Nome immagine senza estensione
        corresponding_masks = [
            os.path.join(masks_dir, mask) for mask in masks if mask.startswith(image_name)
        ]
        if corresponding_masks:
            image_to_masks[os.path.join(images_dir, image)] = corresponding_masks

    # Salvataggio su file JSON
    with open(output_file, 'w') as f:
        json.dump(image_to_masks, f, indent=4)

    print(f"File JSON salvato in: {output_file}")

# Percorsi delle cartelle
masks_dir = "/work/cbaldini/medSAM/code/Genova set/train/masks"
images_dir = "/work/cbaldini/medSAM/code/Genova set/train/images"
json_path = "/work/cbaldini/medSAM/code/Genova set/train/image2label_train.json"

# Generazione del file JSON
generate_image_to_mask_mapping(images_dir, masks_dir, json_path)