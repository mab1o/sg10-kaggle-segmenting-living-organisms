import numpy as np

def extract_ppm_size(ppm_path):
    """
    Extract a size from a PPM image
    Arguments:
        - ppm_path: the path to the PPM image
    Returns:
        - size: the size of ppm
    """
    # Read the PPM image and extract the patch
    with open(ppm_path, "rb") as f:
        # Skip the PPM magic number
        f.readline()
        # Skip the PPM comment
        while True:
            line = f.readline().decode("utf-8")
            if not line.startswith("#"):
                break
        ncols, nrows = map(int, line.split())
    return [ncols, nrows]


def extract_patch_from_ppm(ppm_path, row_idx, col_idx, patch_size):
    """
    Extract a patch from a PPM image
    Arguments:
        - ppm_path: the path to the PPM image
        - row_idx: the row index of the patch
        - col_idx: the column index of the patch
        - patch_size: the size of the patch (width, height)
    Returns:
        - patch: the extracted patch
    """
    # Read the PPM image and extract the patch
    with open(ppm_path, "rb") as f:
        # Skip the PPM magic number
        f.readline()
        # Skip the PPM comment
        while True:
            line = f.readline().decode("utf-8")
            if not line.startswith("#"):
                break
        ncols, nrows = map(int, line.split())
        maxval = int(f.readline().decode("utf-8"))
        
        # Maxval is either lower than 256 or 65536
        # It is actually 255 for the scans, and 65536 for the masks
        # This maximal value impacts the number of bytes used for encoding the pixels' value
        if maxval == 255:
            nbytes_per_pixel = 1
            dtype = np.uint8
        elif maxval == 65535:
            nbytes_per_pixel = 2
            dtype = np.dtype("uint16")
            # The PPM image is in big endian # encoding convention
            dtype = dtype.newbyteorder(">")
        else:
            raise ValueError(f"Unsupported maxval {maxval}")

        first_pixel_offset = f.tell()
        f.seek(0, 2)  # Seek to the end of the file
        data_size = f.tell() - first_pixel_offset
        # Check that the file size is as expected
        assert data_size == (ncols * nrows * nbytes_per_pixel)

        f.seek(first_pixel_offset)  # Seek back to the first pixel

        # Read all the rows of the patch from the image
        patch_size=(patch_size[1],patch_size[0])
        patch = np.zeros(patch_size, dtype=dtype)
        for i in range(patch_size[0]):
            f.seek(
                first_pixel_offset
                + ((row_idx + i) * ncols + col_idx) * nbytes_per_pixel,
                0,  # whence
            )
            row_data = f.read(patch_size[1] * nbytes_per_pixel)
            patch[i] = np.frombuffer(row_data, dtype=dtype)

    return patch
