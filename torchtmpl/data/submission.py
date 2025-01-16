# External imports
import torch

# Local imports
from . import encoder


def image_reconstruction(patches, image_size, patch_size):
    """
    Reconstruct a full image from its patches (patches are torch.Tensors).

    Args:
        patches (list of torch.Tensor): List of image patches.
        image_size (tuple): The size of the full image (height, width).
        patch_size (tuple): The size of each patch (height, width).

    Returns:
        torch.Tensor: The reconstructed image.
    """
    width, height  = image_size
    patch_height, patch_width = patch_size

    # Calculate the number of patches in both dimensions (height and width)
    num_patches_x, num_patches_y = find_num_patches(width, height, patch_height, patch_width)

    # Create an empty tensor to hold the reconstructed image
    reconstructed_image = torch.zeros((height,width), dtype=patches[0].dtype)

    patch_index = 0
    for x in range(num_patches_x):
        for y in range(num_patches_y):
            # Get the current patch from the list
            patch = patches[patch_index]

            x_start, y_start, x_end, y_end, patch_x_start, patch_y_start = find_start_end_coordinate(
                                    x, y, width, height, patch_width,patch_height)

            # Place the patch in the appropriate location
            reconstructed_image[x_start:x_end,y_start:y_end] = patch[patch_x_start:,patch_y_start:]

            # Move to the next patch
            patch_index += 1

    return reconstructed_image


def find_num_patches(width, height, patch_height, patch_width):
    """
    Find the numbers of patches on a image

    Args :
        width, height: size of the image
        patch_width, patch_height: size of the patch
    return :
        num_patches_x, num_patches_y: number of patches on the image
    """
    num_patches_x = height // patch_height
    num_patches_y = width // patch_width
    if height % patch_height != 0:
        num_patches_x += 1
    if width % patch_width != 0:
        num_patches_y += 1
    return num_patches_x,num_patches_y


def find_start_end_coordinate(x, y, width, height, patch_width, patch_height):
    """
    Find the position coordinates

    Args :
        x, y: patch position on the image
        width, height: size of the image
        patch_width, patch_height: size of the patch
    return :
        x_start, y_start: start coordinate of the patch added on the image
        x_end, y_end: end coordinate of the patch added on the image
        patch_x_start, patch_y_start: start coordinate on the patch
    """
    x_start = x * patch_height
    y_start = y * patch_width

    x_end = min((x + 1) * patch_height, height)
    y_end = min((y + 1) * patch_width, width)

    patch_x_start = ((x + 1)*patch_height) - x_end
    patch_y_start = ((y + 1)*patch_width ) - y_end
    
    return x_start, y_start, x_end, y_end, patch_x_start, patch_y_start


def generate_submission_file(predictions, file_name = "submission.csv"):
    """
    Generate the CSV file for kaggle submission

    Args :
        predictions (array of torch.Tensor): table of predicted value, one per image
        file_name (string): name of the submission file
    """
    with open(file_name, "w") as f:
        f.write("Id,Target\n")

        # Let us generate the predictions file for 2 images
        for mask_id in range(len(predictions)):

            # Iteratate over the rows of the prediction
            for idx_row, row in enumerate(predictions[mask_id]):
                row_str = encoder.array_to_string(row.numpy())
                f.write(f"{mask_id}_{idx_row},\"{row_str}\"\n")
   