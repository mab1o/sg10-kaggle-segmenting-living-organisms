# Standard imports
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset

# Local imports
from . import patch
from . import submission

class PlanktonDataset(Dataset):
    def __init__(self, image_mask_dir, patch_size, mode = "train"):
        """
        Args:
            image_mask_dir (str): Path to the directory containing images and mask
            patch_size ([int,int]): Size of the square patch to extract. (width, height)
            mode(str): train or test
        """
        assert mode in ["train", "test"], "mode must be either 'train' or 'test'"
        assert patch_size[0] > 0 and patch_size[1] > 0 , "patch_size should be superior to 0."

        self.mode = mode
        self.image_mask_dir = image_mask_dir
        self.patch_size = patch_size
        self.image_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_scan.png.ppm')])

        # Get mask for train
        if mode == 'train':
            self.mask_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_mask.png.ppm')])
            assert len(self.image_files) == len(self.mask_files), "Mismatch between image and mask files"
        else :
            self.mask_files = []
        
        # Precompute the number of patches for each image
        self.image_patches = []
        self.images_size   = []
        for image_file in self.image_files:
            image_path = os.path.join(self.image_mask_dir, image_file)
            width, height = patch.extract_ppm_size(image_path)
            self.images_size.append((width,height))

            # Add numbers of patchs for the image
            self._add_num_patchs(width,height,patch_size)

        # Calculate the total number of patches
        self.total_patches = sum(x * y for x, y in self.image_patches)


    def _add_num_patchs(self, width, height, patch_size):
        """
        Args:
            width, height: size of the image
            patch_size: size of the patch
        """
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
        assert idx < len(self), f"idx out of range"

        # Determine which image the idx belongs to
        current_idx = idx
        for image_idx, (num_patches_x, num_patches_y) in enumerate(self.image_patches):
            num_patches = num_patches_x * num_patches_y
            if current_idx < num_patches:
                break
            current_idx -= num_patches

        # Determine start row and column index
        image_row, image_column = self._find_row_column(image_idx, current_idx)

        # Load the image and mask
        image_path = os.path.join(self.image_mask_dir, self.image_files[image_idx])
        mask_path  = os.path.join(self.image_mask_dir, self.mask_files[image_idx])

        # Extract the patch using the external function
        image_patch = patch.extract_patch_from_ppm(image_path, image_row, image_column, self.patch_size)
        image_patch = torch.from_numpy(image_patch) #.unsqueeze(0).float()

        if (self.mode == 'test'):
            return image_patch
  
        return image_patch, self._get_mask_for_train(mask_path, image_row, image_column)


    def _get_mask_for_train(self, mask_path, image_row, image_column):
        """
        Args:
            mask_path (str): Path to the directory containing images and mask
            image_row, image_column: image coordinates

        Returns:
            torch.Tensor of the mask
        """
        mask_patch  = patch.extract_patch_from_ppm(mask_path, image_row, image_column, self.patch_size)
        mask_patch  = mask_patch.byteswap().view(mask_patch.dtype.newbyteorder())

        # 0-7 non living == 0; 8-134 living == 1
        mask_patch = np.where(mask_patch < 8, 0, 1)
        mask_patch  = torch.from_numpy(mask_patch)  #.long()

        return mask_patch


    def _find_row_column(self, image_idx, current_idx):
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


    def insert(self, mask_patch, idx = None):
        """
        Insert a mask patch to the Dataset

        Args:
            mask_patch (torch.Tensor): patch of a mask
            idx (int): index of where the patches is added
        """
        assert self.mode == 'test', "Dataset must be a test dataset to insert a patch"
        if idx != None :
            assert 0 < idx < len(self), "Index out of range"

        if idx == None : 
            self.mask_files.append(mask_patch)
        else :
            self.mask_files[idx] = mask_patch


    def show_plankton_patch_image(self, idx, image_name ="plankton_sample.png"):
        """
        Show patch of the plakton image at index idx

        Args:
            mask_patch (torch.Tensor): patch of a mask
            image_name (string): name of the picture
        """
        if(self.mode == 'train'):
            img, mask = self[idx]
        else:
            img = self[idx]
            mask = self.mask_files[idx]
        patch.show_plankton_image(img, mask, image_name)


    def show_plankton_complete_image(self, idx, image_name ="plankton_sample.png"):
        """
        Show complete image of the plakton image at index idx

        Args:
            mask_patch (torch.Tensor): patch of a mask
            image_name (string): name of the picture
        """
        img = patch.extract_patch_from_ppm(
            self.image_mask_dir + self.image_files[idx], 
            0, 0, 
            self.images_size[idx])
        mask = self.reconstruct_mask(idx)
        patch.show_plankton_image(img, mask, image_name)
    

    def to_submission(self,  file_name = "submission.csv") :
        """
        Rebuild mask and write it in a CSV file

        Args :
            data (PlanktonDataset): Dataset of the images
            patches (list of torch.Tensor): mask patches to be rebuild  
            file_name(string): file name at "submission.csv" per default
        """
        assert self.mode == 'test', "to_submission must be use for test dataset"

        # Rebuild mask
        predictions = []
        for image_id in range(len(self.image_files)):
            predictions.append(self.reconstruct_mask(image_id))
        
        # Write Mask in csv file
        submission.generate_submission_file(predictions, file_name)


    def reconstruct_mask(self, image_id):
        """
        Rebuild mask and write it in a CSV file

        Args :
            image_id(int): index of the image mask
        
        Return:
            torch.Tensor of complete mask
        """
        # Find start and end
        id = 0
        start_patche_id = 0
        end_patche_id = self.image_patches[id][0]*self.image_patches[id][1]
        
        while id != image_id :
            id += 1
            start_patche_id = end_patche_id
            end_patche_id += self.image_patches[id][0]*self.image_patches[id][1]
        
        # Reconstruct mask
        image_size = self.images_size[image_id]

        if self.mode == 'test':
            patches = self.mask_files[start_patche_id:end_patche_id]
        else:
            patches = [self[idx_mask][1] for idx_mask in range(start_patche_id,end_patche_id)]
        
        prediction = submission.image_reconstruction(patches, image_size, self.patch_size)
        return prediction


    def get_num_image(self):
        return len(self.image_files)
