# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
import numpy as np

# Local imports
from . import encoder
from . import patch
from . import PlanktonDataset

def test_patch(config) :
    # Test patch
    logging.info("= Test on patch")
    dir_path = config['data']['trainpath'] 

    img_path = dir_path + 'rg20091216_scan.png.ppm'
    logging.info(f"taille de l'image : {patch.extract_ppm_size(img_path)}")
    img  = patch.extract_patch_from_ppm(img_path, 4000, 4000, [1240,1240])

    mask_path = dir_path + 'rg20091216_mask.png.ppm'
    logging.info(f"taille du mask : {patch.extract_ppm_size(img_path)}")
    mask = patch.extract_patch_from_ppm(mask_path, 4000, 4000, [1240,1240])

    patch.show_plankton_image(img,mask)
    logging.info("= FIN Test on patch")

def test_max_patch_size_working(config) :

    logging.info("= Test on patch")
    dir_path = config['data']['trainpath'] 

    max_size_not_found = True
    max_size = 1730
    while(max_size_not_found) :

        try: 
            img_path = dir_path + 'rg20090114_scan.png.ppm'
            logging.info(f"taille de l'image : {patch.extract_ppm_size(img_path)}")
            img  = patch.extract_patch_from_ppm(img_path, 12806, 0, (max_size,max_size))

            mask_path = dir_path + 'rg20090114_mask.png.ppm'
            logging.info(f"taille du mask : {patch.extract_ppm_size(img_path)}")
            mask = patch.extract_patch_from_ppm(mask_path, 12806, 0, (max_size,max_size))

            logging.info(f"patch size of {max_size} work !!!")
            max_size_not_found = False

        except Exception as e:
            logging.info(f"patch size of {max_size} doesn't work.")
            max_size -= 1

    logging.info("= FIN Test on patch")

def test_encoder(config):
    # Test encoder
    logging.info("= Test the encoder")
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


def test_PlanktonDataset(config):
    logging.info("=Test Plankton Dataset")
    dir_path = config['data']['trainpath'] 
    data = PlanktonDataset(dir_path,(10000,10000))

    logging.info(f"\ntaille du dataset : {len(data)}")
    logging.info(f"\nPath d'origin des images : {data.image_mask_dir}")
    logging.info(f"\nTaille du patch : {data.patch_size }")
    logging.info(f"\nFichiers images extraits : {data.image_files}")
    logging.info(f"\nFichier mask extrait : {data.mask_files}")
    logging.info(f"\nPathes creer sur chaque image : {data.image_patches}")
    logging.info(f"\nTaille respective de chaque image : {data.images_size}")
    
    logging.info(f"\nImpression des patchs")
    num_patches_first_image = data.image_patches[0][0] * data.image_patches[0][1]
    for i in range(num_patches_first_image):
        logging.info(f"Processing index {i}...")
        try:
            image, mask = data[i]
            patch.show_plankton_image(image, mask, f"image_{i}")
        except Exception as e:
            logging.info(f"Error at index {i}: {e}")

    logging.info(f"\nImpression de l'image original")
    dir_path = config['data']['trainpath']
    img  = patch.extract_patch_from_ppm(dir_path + data.image_files[0], 0, 0, (14529,22807))
    mask = patch.extract_patch_from_ppm(dir_path + data.mask_files[0], 0, 0, (14529,22807))
    patch.show_plankton_image(img, mask, f"image_original")

    logging.info("= FIN Test on Dataset")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage : {sys.argv[0]} config.yaml ")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))
    
    # test_encoder(config)
    # test_patch(config)
    # test_max_patch_size_working(config)
    test_PlanktonDataset(config)
