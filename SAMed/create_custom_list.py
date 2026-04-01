import os

# Set the folder to read files from
input_folder = "/work/cbaldini/medSAM/code/Genova set/hennce"  # Change this to your folder path
output_file = '/work/cbaldini/medSAM/code/SAMed/lists/lists_GenovaHENANCE/train.txt'  # Change this to your desired output file path
# input_folder = "/work/cbaldini/medSAM/code/Genova set/henance/easy_margins"  # Change this to your folder path
# output_file = "/work/cbaldini/medSAM/code/Genova set/test/test.txt"  # Change this to your desired output file path
# input_folder = "/work/cbaldini/medSAM/code/laryngoscope_dataset/train"  # Change this to your folder path
# output_file = "/work/cbaldini/medSAM/code/SAMed/lists/lists_laryngoscope/train.txt"  # Change this to your desired output file path
# input_folder = "/work/cbaldini/medSAM/code/laryngoscope_dataset/test"  # Change this to your folder path
# output_file = "/work/cbaldini/medSAM/code/SAMed/lists/lists_GenovaHENANCE/train_easy.txt"  # Change this to your desired output file path

# # Open the output file in write mode
# with open(output_file, 'w') as f:
#     # Walk through the directory
#     for root, _, files in os.walk(input_folder):
#         for file in sorted(files):  # Sort files by name
            # if '.npz' in file:
            #     # Write each file name to the output file, one per line
            #     f.write(f"{file.split('.npz')[0]}\n")
            
# # Open the output file in write mode
# with open(output_file, 'w') as f:
#     # Walk through the directory
#     for file in os.listdir(os.path.join(input_folder, 'imgs')):
#         if '.npy' in file:
#                 # Write each file name to the output file, one per line
#                 f.write(f"{file.split('.npy')[0]}\n")

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Walk through the directory
    for file in os.listdir(os.path.join(input_folder, 'images')):
        if '.png' in file:
                # Write each file name to the output file, one per line
                f.write(f"{file.split('.png')[0]}\n")

print(f"File successfully generated: {output_file}")