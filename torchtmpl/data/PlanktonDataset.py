# Standard imports
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset

# Local imports
from . import patch

class PlanktonDataset(Dataset):
    def __init__(self, image_mask_dir, patch_size):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            patch_size ([int,int]): Size of the square patch to extract.
        """
        self.image_mask_dir = image_mask_dir
        self.patch_size = patch_size
        self.image_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_scan.png.ppm')])
        self.mask_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_mask.png.ppm')])

        assert len(self.image_files) == len(self.mask_files), "Mismatch between image and mask files"

        # Precompute the number of patches for each image
        self.image_patches = []
        self.images_size   = []
        for image_file in self.image_files:
            image_path = os.path.join(self.image_mask_dir, image_file)
            width, height = patch.extract_ppm_size(image_path)
            self.images_size.append((width,height))

            # Add numbers of patchs for the image
            self.add_num_patchs(width,height,patch_size)

        # Calculate the total number of patches
        self.total_patches = sum(x * y for x, y in self.image_patches)

    def add_num_patchs(self,width,height, patch_size):
        # Check patch possible
        assert width  > patch_size[0], f" Impossible la patch est plus grand que l'image {patch_size[0]} > {width}"
        assert height > patch_size[1], f" Impossible la patch est plus grand que l'image {patch_size[1]} > {height}"

        num_patches_x = width // patch_size[0]
        if width % patch_size[0] != 0 :
            num_patches_x += 1
        
        num_patches_y = height // patch_size[1]
        if height % patch_size[1] != 0 :
            num_patches_y += 1

        self.image_patches.append((num_patches_x, num_patches_y))

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Global index of the patch.

        Returns:
            patch (torch.Tensor): Image patch of size (C, patch_size[0], patch_size[1]).
            mask_patch (torch.Tensor): Corresponding mask patch.
        """
        # Check idx exist
        if idx > len(self):
            return None, None

        # Determine which image the idx belongs to
        current_idx = idx
        for image_idx, (num_patches_x, num_patches_y) in enumerate(self.image_patches):
            num_patches = num_patches_x * num_patches_y
            if current_idx < num_patches:
                break
            current_idx -= num_patches

        # Determine start row and column index
        image_row, image_column = self.find_row_column(image_idx, current_idx)

        # Load the image and mask
        image_path = os.path.join(self.image_mask_dir, self.image_files[image_idx])
        mask_path  = os.path.join(self.image_mask_dir, self.mask_files[image_idx])

        # Extract the patch using the external function
        image_patch = patch.extract_patch_from_ppm(image_path, image_row, image_column, self.patch_size)
        mask_patch  = patch.extract_patch_from_ppm(mask_path, image_row, image_column, self.patch_size)
        mask_patch  = mask_patch.byteswap().view(mask_patch.dtype.newbyteorder())

        # 0-7 non living == 0; 8-134 living == 1
        mask_patch = np.where(mask_patch < 8, 0, 1)

        # Convert to PyTorch tensors
        image_patch = torch.from_numpy(image_patch) #.unsqueeze(0).float()
        mask_patch  = torch.from_numpy(mask_patch)  #.long()
  
        return image_patch, mask_patch
    
    def find_row_column(self, image_idx, current_idx):
        """
        Args:
            image_idx (int): index of the image.
            patch_x (int): index of the row patch
            patch_y (int): index of the column patch

        Returns:
            image_row (int): index of a column in the image.
            image_column (int): index of a column in the image.
        """

        width, height = self.images_size[image_idx]
        num_patches_x, num_patches_y = self.image_patches[image_idx]
        patch_width, patch_height = self.patch_size
        
        # Local patch index
        patch_x = current_idx % num_patches_x
        patch_y = current_idx // num_patches_x

        # Local start row and column index
        if (patch_y == (num_patches_y-1)):
            image_row = height - patch_height
        else:
            image_row = patch_height * patch_y
        
        if (patch_x == (num_patches_x-1)):
            image_column= width - patch_width
        else :
            image_column = patch_width * patch_x
        
        return image_row, image_column
