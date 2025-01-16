from .dataloader import get_dataloaders
from .planktonds import PlanktonDataset

"""
__main__: test file
    - test_encoder(config)
    - test_patch(config)
    - test_PlanktonDataset_train(config)
    - test_generate_csv_file(config)
    - test_reconstruction_image_test(config)
    - test_compare_image_test(config)

data: data management
    - show_image(X)
    - get_dataloaders(data_config, use_cuda):

PlanktonDataset: dataset inherited from torch.Dataset
    - __init__(self, image_mask_dir, patch_size, mode = "train")
    - __len__(self)
    - __getitem__(self, idx)
    - insert(self, mask_patch, idx = None)
    - show_plankton_patch_image(self, idx, image_name ="plankton_sample.png")
    - show_compare_mask(self, idx, real_dataset, image_name = "compare_mask.png")
    - to_submission(self,  file_name = "submission.csv")
    - reconstruct_mask(self, image_id)
    - get_num_image(self)
    - show_plankton_complete_image(self, idx, image_name ="plankton_sample.png")

encoder: utils for binary to string
    - def generate_sample_files(img_height, img_width)
    - array_to_string(arr: np.array, num_bits=6, offset=48)
    - binary_list_to_string(binary_list, num_bits=6, offset=48)

patch: utils to extract patch from a image
    - show_plankton_image(img, mask, image_name = "plankton_sample.png")
    - extract_ppm_size(ppm_path)
    - extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size)

submission: utils for submission
    - image_reconstruction(patches, image_size, patch_size)
    - find_num_patches(width, height, patch_height, patch_width)
    - generate_submission_file(predictions, file_name = "submission.csv")
    - to_submission(data : PlanktonDataset, patches,  file_name = "submission.csv")
"""
