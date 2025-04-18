import numpy as np
import torch


def generate_submission_file(
    predictions: list[torch.Tensor],
    ordered_files: list[str],
    file_name="submission.csv",
):
    """Generate the CSV file for kaggle submission.

    Args :
        predictions (array of torch.Tensor): table of predicted value, one per image
        ordered_files (array of string): table of the name of the files
        file_name (string): name of the submission file

    """
    assert len(ordered_files) == len(predictions), (
        "The list of file name must be the same size as the list of prediction"
    )

    with open(file_name, "w") as f:
        f.write("Id,Target\n")

        # Let us generate the predictions file for 2 images
        for mask_id, mask_file_name in enumerate(ordered_files):
            # Iteratate over the rows of the prediction
            for idx_row, row in enumerate(predictions[mask_id]):
                row = row.cpu().numpy().astype(int)
                row_str = array_to_string(row)
                f.write(f'{mask_file_name}_{idx_row},"{row_str}"\n')


def binary_list_to_string(binary_list, num_bits=6, offset=48):
    """Convert a list of binary digits (0s and 1s) into a string, where every 6 bits represent a character.

    Args:
        binary_list: List of integers (0 or 1) representing binary digits.
        num_bits: Number of bits to use for encoding.
        offset: Offset to add to the integer representation of the binary list.

    Returns:
        String representation of the binary input.

    """
    # Ensure the binary list length is a multiple of num_bits
    if len(binary_list) % num_bits != 0:
        raise ValueError(f"The binary list length must be a multiple of {num_bits}.")

    # Split the list into chunks of num_bits
    chars = []
    for i in range(0, len(binary_list), num_bits):
        byte = binary_list[i : i + num_bits]
        # Convert the byte (list of num_bits bits) to an integer
        byte_as_int = offset + int("".join(map(str, byte)), 2)
        # Convert the integer to a character and append to the result list
        chars.append(chr(byte_as_int))

    return "".join(chars)


def array_to_string(arr: np.array, num_bits=6, offset=48):
    """Transform array of 0 and 1 to a ASCII string.

    Args:
        arr: a nd array of 0's and 1's
        num_bits: number of bits to use for encoding
        offset: offset to add to the integer representation of the binary list

    Returns:
        String representation of the binary input.

    """
    raveled = list(arr.ravel())
    # Pad the raveled sequence by 0's to have a size multiple of num_bits
    raveled.extend([0] * (num_bits - (len(raveled) % num_bits)))
    result = binary_list_to_string(raveled, num_bits, offset)
    return result


def generate_sample_files(img_height, img_width):
    """Generate sample image convert into csv file.

    Args:
        img_height: image height
        img_width: image width

    Returns:
        None

    """
    with open("submission.csv", "w") as f:
        f.write("Id,Target\n")

        # Let us generate the predictions file for 2 images
        for mask_id in range(2):
            # For this example, we generate a random prediction
            prediction = np.random.randint(0, 2, (img_height, img_width))

            # Iteratate over the rows of the prediction
            for idx_row, row in enumerate(prediction):
                mystr = array_to_string(row)
                f.write(f'{mask_id}_{idx_row},"{mystr}"\n')


def generate_sample_files_unbalanced(
    img_height, img_width, proportion_zeros=0.9844, proportion_ones=0.0156
):
    """Generate a sample image and convert it into a CSV file with specific proportions of 0s and 1s.

    Args:
        img_height: image height
        img_width: image width
        proportion_zeros: proportion of 0s in the mask
        proportion_ones: proportion of 1s in the mask
    Returns:
        None

    """
    with open("submission.csv", "w") as f:
        f.write("Id,Target\n")

        # Let us generate the predictions file for 2 images
        for mask_id in range(2):
            # Generate a random prediction with specified proportions
            total_pixels = img_height * img_width
            num_ones = int(total_pixels * proportion_ones)
            num_zeros = total_pixels - num_ones

            # Create an array with the desired proportions
            flat_mask = np.array([1] * num_ones + [0] * num_zeros)
            np.random.shuffle(flat_mask)  # Shuffle to randomize the arrangement
            prediction = flat_mask.reshape((img_height, img_width))

            # Iterate over the rows of the prediction
            for idx_row, row in enumerate(prediction):
                mystr = array_to_string(row)
                f.write(f'{mask_id}_{idx_row},"{mystr}"\n')
