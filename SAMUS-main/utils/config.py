# This file is used to configure the training parameters for each task

class Config_ENDOGE:
    # This dataset is for breast cancer segmentation
    data_path = "/work/cbaldini/medSAM/code/Seoul set" # "/work/cbaldini/medSAM/code/Genova set/henance"
    data_subpath = "/work/cbaldini/medSAM/code/Seoul set" # "/work/cbaldini/medSAM/code/Genova set/henance"   
    save_path = "./checkpoints/AutoSAMUS_SE/"
    result_path = "./result/AutoSAMUS_SEtestcontour/"
    # tensorboard_path = "./tensorboard/BUSI/"
    load_path = "/work/cbaldini/medSAM/code/SAMUS-main/checkpoints/AutoSAMUS_GE_improved/AutoSAMUS_07300857_219_0.7309735024213018.pth" #/work/cbaldini/medSAM/code/SAMUS-main/checkpoints/GE_400ep/SAMUS__399.pth" #"/work/cbaldini/medSAM/code/sam_vit_b_01ec64.pth"  /work/cbaldini/medSAM/code/MedSAM/medsam_vit_b.pth" #/work/cbaldini/medSAM/code/sam_vit_b_01ec64.pth" #"/work/cbaldini/medSAM/code/SAMUS-main/checkpoints/GEauto/AutoSAMUS__99.pth" #"/work/cbaldini/medSAM/code/SAMUS-main/checkpoints/GE/SAMUS__99.pth" # "/work/cbaldini/medSAM/code/sam_vit_b_01ec64.pth" #save_path + "/xxx.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 10                        # number of total epochs to run (default: 400)
    batch_size = 2                     # batch size (default: 4), was 8
    learning_rate = 0.005                # initial learning rate (default: 0.001), was 0.005
    # momentum = 0.9                      # momentum
    classes = 2                         # the number of classes (background + foreground)
    img_size = 256                      # the input size of model (was 256)
    num_points = 10                     # the number of points to sample from the mask
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    # test_split = "100 LAR selected"     # the file name of testing set
    # test_split = "henance"
    test_split = "all"
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "no"                        # the type of input image
    img_channel = 3                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = True
    mode = "train"
    visual = False
    modelname = "AutoSAMUS"  # was SAMUS

# ==================================================================================================
def get_config(task="ENDOGE"):
    if task == "ENDOGE":
        return Config_ENDOGE()
    else:
        assert("We do not have the related dataset, please choose another task.")

