# Standard imports
import logging
import sys

# External imports
import yaml
import numpy as np

# Local imports
from . import patch
from . import PlanktonDataset
from . import encoder
from . import submission

def test_patch(config) :
    # Test patch
    logging.info("\n= Test on patch")
    dir_path = config['data']['trainpath'] 

    img_path = dir_path + 'rg20091216_scan.png.ppm'
    logging.info(f"taille de l'image : {patch.extract_ppm_size(img_path)}")
    img  = patch.extract_patch_from_ppm(img_path, 4000, 4000, [1240,1240])

    mask_path = dir_path + 'rg20091216_mask.png.ppm'
    logging.info(f"taille du mask : {patch.extract_ppm_size(img_path)}")
    mask = patch.extract_patch_from_ppm(mask_path, 4000, 4000, [1240,1240])

    patch.show_plankton_image(img,mask)
    logging.info("= FIN Test on patch")


def test_PlanktonDataset_train(config):
    logging.info("\n=Test Plankton Dataset")
    dir_path = config['data']['trainpath'] 
    data = PlanktonDataset(dir_path,(10000,10000))

    logging.info(f"\nDataset size: {len(data)}")
    logging.info(f"\nOrigin Path: {data.image_mask_dir}")
    logging.info(f"\nPatch size: {data.patch_size }")
    logging.info(f"\nImage files: {data.image_files}")
    logging.info(f"\nMask files: {data.mask_files}")
    logging.info(f"\nPathes per image : {data.image_patches}")
    logging.info(f"\nSize of each image: {data.images_size}")
    
    logging.info(f"\nImpression des patchs")
    num_patches_first_image = data.image_patches[0][0] * data.image_patches[0][1]
    for i in range(num_patches_first_image):
        logging.info(f"Processing index {i}...")
        try:
            data.show_plankton_patch_image(i, f"image_{i}")
        except Exception as e:
            logging.info(f"Error at index {i}: {e}")

    logging.info(f"\nImpression de l'image original")
    dir_path = config['data']['trainpath']
    img  = patch.extract_patch_from_ppm(dir_path + data.image_files[0], 0, 0, (22807,14529))
    mask = patch.extract_patch_from_ppm(dir_path + data.mask_files[0], 0, 0, (22807,14529))
    patch.show_plankton_image(img, mask, f"image_original")

    logging.info("= FIN Test on Dataset")


def test_encoder(config):
    # Test encoder
    logging.info("\n= Test the encoder")
    binary_to_encode = [
        "1111",
        "111100",
        "101101001010011000111001000011011010100010001010110110100011111100001101000000",
        "111000100100000000100101111011010001010111101000001100110000001101011010101000"
    ]
    expected_results = [
        "l",
        "l",
        "]:Hi3JR:fSl=0",
        "hT0UkAGX<`=JX" 
    ]
    for binary, expected_result in zip(binary_to_encode,expected_results):
        result = encoder.array_to_string(np.array(list(binary)))
        if result == expected_result :
            logging.info(f"{binary} has been correctly encode to {expected_result}.")
        else :
            logging.error(f"{binary} has been incorrectly encode to {result} instead of {expected_result}.")
    logging.info("= FIN Test on encoder")


def test_reconstruction_image(config):
    logging.info("\n=Test Submission reconstruct")

    logging.info("\nCharge Data")
    dir_path = config['data']['trainpath']
    patch_size = (10000,10000)
    ds = PlanktonDataset(dir_path,patch_size)
    
    logging.info(f"\nImpression de l'image original")
    dir_path = config['data']['trainpath']
    img  = patch.extract_patch_from_ppm(dir_path + ds.image_files[1], 0, 0, ds.images_size[1])
    mask = patch.extract_patch_from_ppm(dir_path + ds.mask_files[1], 0, 0, ds.images_size[1])
    patch.show_plankton_image(img, mask, "image_original_2")

    logging.info(f"\nImpression de l'image reconstruite")
    ds.show_plankton_complete_image(1, "image_reconstruct")

    logging.info("\n=FIN Test Submission reconstruct")


def test_generate_csv_file(config):
    logging.info("\n=Test Submission formation du mask")

    logging.info("\nCharge Data")
    dir_path = config['data']['trainpath']
    patch_size = (20000,2)
    ds = PlanktonDataset(dir_path,patch_size)

    logging.info("\nMake csv file")
    _, mask = ds[10000]
    _, mask1 = ds[11000]
    print(mask)
    submission.generate_submission_file([mask, mask1])

    logging.info("\n= FIN Test Submission formation du mask")

def test_reconstruction_image_test(config):
    logging.info("\n=Test Submission reconstruct test")

    logging.info("\nCharge Data train")
    dir_path = config['data']['trainpath']
    patch_size = (10000,10000)
    ds_train = PlanktonDataset(dir_path,patch_size)

    logging.info("\nCharge Data test")
    ds_test = PlanktonDataset(dir_path,patch_size, mode='test')

    num_patches_first_image = ds_train.image_patches[0][0] * ds_train.image_patches[0][1]
    for i in range(num_patches_first_image):
        logging.info(f"Processing index {i}...")
        try:
            ds_test.insert(ds_train[i][1])
        except Exception as e:
            logging.info(f"Error at index {i}: {e}")

    logging.info(f"\nImpression de l'image original")
    dir_path = config['data']['trainpath']
    img  = patch.extract_patch_from_ppm(dir_path + ds_train.image_files[0], 0, 0, ds_train.images_size[0])
    mask = patch.extract_patch_from_ppm(dir_path + ds_train.mask_files[0], 0, 0, ds_train.images_size[0])
    # patch.show_plankton_image(img, mask, "image_original")

    logging.info(f"\nImpression de l'image reconstruite")
    ds_test.show_plankton_complete_image(0, "image_test_reconstruct")

    logging.info("\n=FIN Test Submission reconstruct test")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage : {sys.argv[0]} config.yaml ")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))
    
    #test_encoder(config)
    #test_patch(config)
    #test_encoder(config)
    #test_generate_csv_file(config)
    #test_PlanktonDataset_train(config)
    #test_reconstruction_image(config)
    test_reconstruction_image_test(config)
