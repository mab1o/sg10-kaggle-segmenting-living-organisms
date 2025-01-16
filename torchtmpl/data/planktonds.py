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
        self.categories = ['non living','living']
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
        
        num_patches_x = -(-width // patch_size[0]) 
        num_patches_y = -(-height // patch_size[1])

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
        assert idx < len(self), f"Index out of range: {idx}"

        # Determine which image the idx belongs to
        current_idx = idx
        for image_idx, (num_patches_x, num_patches_y) in enumerate(self.image_patches):
            num_patches = num_patches_x * num_patches_y
            if current_idx < num_patches:
                break
            current_idx -= num_patches

        # Determine start row and column index
        image_row, image_column = self._find_row_column(image_idx, current_idx)

        # Load the image
        image_path = os.path.join(self.image_mask_dir, self.image_files[image_idx])
        image_patch = patch.extract_patch_from_ppm(image_path, image_row, image_column, self.patch_size)
        image_patch = torch.from_numpy(image_patch).unsqueeze(0).float()

        if (self.mode == 'test'):
            return image_patch
  
        return image_patch, self._get_mask_for_train(image_idx, image_row, image_column)


    def _get_mask_for_train(self, image_idx, image_row, image_column):
        """
        Load the mask

        Args:
            mask_path (str): Path to the directory containing images and mask
            image_row, image_column: image coordinates

        Returns:
            torch.Tensor of the mask
        """
        mask_path  = os.path.join(self.image_mask_dir, self.mask_files[image_idx])
        mask_patch  = patch.extract_patch_from_ppm(mask_path, image_row, image_column, self.patch_size)
        mask_patch  = mask_patch.byteswap().view(mask_patch.dtype.newbyteorder())

        # Divide living from non living : 0-7 non living == 0; 8-134 living == 1 
        mask_patch = np.where(mask_patch < 8, 0, 1)
        mask_patch  = torch.from_numpy(mask_patch).float()

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
        num_patches_x, _ = self.image_patches[image_idx]
        patch_width, patch_height = self.patch_size
        
        patch_x = current_idx % num_patches_x
        patch_y = current_idx // num_patches_x

        image_row = min(patch_height * patch_y, height - patch_height)  # **(Simplified boundary condition)**
        image_column = min(patch_width * patch_x, width - patch_width) 
        
        return image_row, image_column


    def insert(self, mask_patch, idx = None):
        """
        Insert a mask patch to the Dataset

        Args:
            mask_patch (torch.Tensor): patch of a mask
            idx (int): index of where the patches is added
        """
        assert self.mode == 'test', "Dataset must be a test dataset to insert a patch"

        if idx == None :
            self.mask_files.append(mask_patch)
        else :
            assert 0 <= idx < len(self), "Index out of range: {idx}"
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
            mask = self.mask_files[idx] if idx < len(self.mask_files) else None
        patch.show_plankton_image(img, mask, image_name)


    def show_compare_mask(self, idx, real_dataset , image_name = "compare_mask.png"):
        """
        Show complete image of the plakton image at index idx

        Args:
            idx (int): index of the image
            real_dataset (PlanktonDataset): dataset with real mask
        """
        assert idx < self.get_num_image(), f"Index out of range: {idx}"
        assert isinstance(real_dataset, PlanktonDataset), "real_dataset must be an instance of PlanktonDataset"

        real_mask = patch.extract_patch_from_ppm(
            real_dataset.image_mask_dir + real_dataset.mask_files[idx], 
            0, 0, 
            real_dataset.images_size[idx])
        real_mask = np.where(real_mask < 8, 0, 1)
        
        predict_mask = self.reconstruct_mask(idx)
        predict_mask += real_mask

        patch.show_plankton_mask(real_mask, predict_mask, image_name)


    def show_plankton_complete_image(self, idx, image_name ="plankton_sample.png"):
        """
        Show complete image of the plakton image at index idx

        Args:
            mask_patch (torch.Tensor): patch of a mask
            image_name (string): name of the picture
        """
        assert idx < self.get_num_image(), f"Index out of range: {idx}"

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
        assert file_name.endswith('.csv'), "File name must end with .csv"
        
        predictions = [
            self.reconstruct_mask(image_id) for image_id in range(len(self.image_files))
        ]
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
        start_patche_id, end_patche_id= 0, 0       
        for id in range (image_id + 1):
            start_patche_id = end_patche_id
            end_patche_id  += self.image_patches[id][0]*self.image_patches[id][1]
        
        # Reconstruct mask
        image_size = self.images_size[image_id]

        if self.mode == 'test':
            patches = self.mask_files[start_patche_id:end_patche_id]
        else:
            patches = [self[idx_mask][1] for idx_mask in range(start_patche_id,end_patche_id)]
        
        return submission.image_reconstruction(patches, image_size, self.patch_size)


    def get_num_image(self):
        return len(self.image_files)
