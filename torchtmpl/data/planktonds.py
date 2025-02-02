# Standard imports
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# Local imports
from . import patch
from . import submission

class PlanktonDataset(Dataset):
    def __init__(self, image_mask_dir, patch_size, mode="train", transform=None, apply_transform=True):
        """
        Args:
            image_mask_dir (str): Path to the directory containing images and masks.
            patch_size (tuple): Size of the square patch to extract (width, height).
            mode (str): Either 'train' or 'test'.
            transform (list): List of transformations.
            apply_transform (bool): If True, apply transformations (used for train only).
        """
        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
        assert patch_size[0] > 0 and patch_size[1] > 0, "Patch size must be greater than 0."

        self.mode = mode
        self.transform = transform
        self.apply_transform = apply_transform  # Nouveau paramètre pour désactiver les transformations sur valid
        self.image_mask_dir = image_mask_dir
        self.patch_size = patch_size
        self.image_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_scan.png.ppm')])

        if mode == 'train':
            self.mask_files = sorted([f for f in os.listdir(image_mask_dir) if f.endswith('_mask.png.ppm')])
            assert len(self.image_files) == len(self.mask_files), "Mismatch between image and mask files"
        else:
            self.mask_files = []
        
        # Precompute patch information
        self.image_patches = []
        self.images_size = []
        for image_file in self.image_files:
            image_path = os.path.join(self.image_mask_dir, image_file)
            width, height = patch.extract_ppm_size(image_path)
            self.images_size.append((width, height))
            self.image_patches.append(self._calculate_num_patches(width, height))

        self.total_patches = sum(x * y for x, y in self.image_patches)


            
    def _calculate_num_patches(self, width, height):
        """
        Calculate the number of patches in each dimension for a given image size.
        Args:
            width (int): Width of the image.
            height (int): Height of the image.
        Returns:
            tuple: Number of patches along width and height.
        """
        assert width > self.patch_size[0] and height > self.patch_size[1], "Patch size is larger than the image dimensions."
        num_patches_x = -(-width // self.patch_size[0])
        num_patches_y = -(-height // self.patch_size[1])
        return num_patches_x, num_patches_y

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        assert idx < len(self), f"Index out of range: {idx}"

        # Identifier le patch image
        current_idx = idx
        for image_idx, (num_patches_x, num_patches_y) in enumerate(self.image_patches):
            num_patches = num_patches_x * num_patches_y
            if current_idx < num_patches:
                break
            current_idx -= num_patches

        image_row, image_column = self._find_row_column(image_idx, current_idx)
        image_path = os.path.join(self.image_mask_dir, self.image_files[image_idx])
        image_patch = patch.extract_patch_from_ppm(image_path, image_row, image_column, self.patch_size)

        if self.mode == 'test':
            return torch.from_numpy(image_patch).unsqueeze(0).float()

        mask_patch = self._get_mask_patch(image_idx, image_row, image_column)

        # Appliquer la transformation uniquement si `apply_transform` est True
        if self.apply_transform and self.transform:
            augmented = self.transform(image=image_patch, mask=mask_patch)
            image_patch, mask_patch = augmented["image"], augmented["mask"]

        image_patch = torch.from_numpy(image_patch).unsqueeze(0).float()
        mask_patch = torch.from_numpy(mask_patch).float()

        return image_patch, mask_patch

    def _get_mask_patch(self, image_idx, image_row, image_column):
        """
        Extract the mask patch corresponding to the image patch.
        Args:
            image_idx (int): Index of the image.
            image_row (int): Row index of the patch.
            image_column (int): Column index of the patch.
        Returns:
            torch.Tensor: Mask patch.
        """
        mask_path  = os.path.join(self.image_mask_dir, self.mask_files[image_idx])
        mask_patch = patch.extract_patch_from_ppm(mask_path, image_row, image_column, self.patch_size)
        mask_patch = mask_patch.byteswap().view(mask_patch.dtype.newbyteorder())
        mask_patch = np.where(mask_patch < 8, 0, 1) # non living : 0-7; living : 8-134 
        return mask_patch

    def _find_row_column(self, image_idx, current_idx):
        """
        Calculate the row and column index for a given patch index.
        Args:
            image_idx (int): Index of the image.
            current_idx (int): Patch index within the image.
        Returns:
            tuple: (row index, column index) of the patch.
        """
        width, height = self.images_size[image_idx]
        num_patches_x, _ = self.image_patches[image_idx]
        patch_width, patch_height = self.patch_size
        
        patch_x = current_idx % num_patches_x
        patch_y = current_idx // num_patches_x

        image_row = min(patch_height * patch_y, height - patch_height)
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
        
        if self.mode == 'train':
            name_mask = self.mask_files
        else:
            name_mask = [f.replace('_scan.png.ppm', '_mask.png.ppm') for f in self.image_files]

        predictions = [
            self.reconstruct_mask(image_id) for image_id in range(len(self.image_files))
        ]
        submission.generate_submission_file(predictions, name_mask, file_name)

    def reconstruct_mask(self, idx, binary=True):
        """
        Reconstruct the full mask for a given image.

        Args:
            idx (int): ID of the image.
            binary (bool): If True, applies thresholding (0/1).
                        If False, preserves raw model output.

        Returns:
            torch.Tensor: Reconstructed mask.
        """
        start_idx = sum(x * y for x, y in self.image_patches[:idx])
        end_idx = start_idx + self.image_patches[idx][0] * self.image_patches[idx][1]

        patches = [self[i][1] if self.mode == 'train' else self.mask_files[i]
                for i in range(start_idx, end_idx)]

        width, height = self.images_size[idx]
        patch_width, patch_height = self.patch_size
        num_patches_width, num_patches_height = self.image_patches[idx]

        reconstruct_mask = torch.zeros((height, width), dtype=torch.float32)  # Ensure correct dtype

        patch_index = 0
        for x in range(num_patches_height):
            for y in range(num_patches_width):
                patch = patches[patch_index]

                x_start = x * patch_height
                y_start = y * patch_width
                x_end = min((x + 1) * patch_height, height)
                y_end = min((y + 1) * patch_width, width)

                patch_x_start = ((x + 1) * patch_height) - x_end
                patch_y_start = ((y + 1) * patch_width) - y_end

                # Ensure patch is 2D (H, W)
                if patch.dim() == 4:
                    patch = patch.squeeze(0).squeeze(0)  # Convert (1, 1, H, W) -> (H, W)
                elif patch.dim() == 3:
                    patch = patch.squeeze(0)  # Convert (1, H, W) -> (H, W)

                # Apply binary thresholding if needed (for RegNetY or other models outputting logits)
                if binary:
                    patch = (patch > 0.5).long()

                # Assign to the final reconstructed mask
                reconstruct_mask[x_start:x_end, y_start:y_end] = patch[patch_x_start:, patch_y_start:]

                patch_index += 1

        return reconstruct_mask




    def __repr__(self):
        return f"PlanktonDataset(mode={self.mode},"\
             + f"total_images={len(self.image_files)},"\
             + f"total_patches={self.total_patches})"
