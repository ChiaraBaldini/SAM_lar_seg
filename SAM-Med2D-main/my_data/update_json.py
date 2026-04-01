import os
import json

masks_dir = "/work/cbaldini/medSAM/code/Genova set/train/masks"
images_dir = "/work/cbaldini/medSAM/code/Genova set/train/images"
json_path = "/work/cbaldini/medSAM/code/Genova set/train/label2image_test.json"

# Carica il file json esistente (se esiste)
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        mapping = json.load(f)
else:
    mapping = {}

# Trova tutte le maschere nella cartella
mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for mask_file in mask_files:
    mask_path = f"{masks_dir}/{mask_file}"
    image_path = f"{images_dir}/{mask_file}"
    # Aggiungi solo se esiste anche l'immagine corrispondente
    if os.path.exists(image_path):
        mapping[mask_path] = image_path

# Salva il file json aggiornato
with open(json_path, "w") as f:
    json.dump(mapping, f, indent=4)

print(f"File {json_path} aggiornato con {len(mapping)} coppie maschera/immagine.")