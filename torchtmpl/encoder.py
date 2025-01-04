'''
This script illustrates how to encode your prediction into the expected submission format. 

The generate_sample_files() generates 2 random arrays and write them into the expected submission format 
where ASCII characters encodes the predictions of 6 consecutive pixels.
'''

# External imports
import numpy as np

def binary_list_to_string(binary_list, num_bits=6, offset=48):
    """
    Convert a list of binary digits (0s and 1s) into a string, where every 6 bits represent a character.

    Arguments:
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
    """
    Arguments:
        arr: a nd array of 0's and 1's
        num_bits: number of bits to use for encoding
        offset: offset to add to the integer representation of the binary list

    Returns:
        String representation of the binary input.
    """
    raveled = list(arr.ravel())
    # Pad the raveled sequence by 0's to have a size multiple of num_bits
    raveled.extend([0] * ((num_bits - (len(raveled) % num_bits)) % num_bits))
    result = binary_list_to_string(raveled, num_bits, offset)
    return result


def generate_sample_files(img_height, img_width):
    with open("submission.csv", "w") as f:
        f.write("Id,Target\n")

        # Let us generate the predictions file for 2 images
        for mask_id in range(2):
            # For this example, we generate a random prediction
            prediction = np.random.randint(0, 2, (img_height, img_width))

            # Iteratate over the rows of the prediction
            for idx_row, row in enumerate(prediction):
                mystr = array_to_string(row)
                f.write(f"{mask_id}_{idx_row},\"{mystr}\"\n")
