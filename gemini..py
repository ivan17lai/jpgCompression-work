# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn, idctn # Added idctn for completeness if needed later
import time
import os # Added for file path handling

# Ensure the 'file' directory exists or adjust paths accordingly
if not os.path.exists('file'):
    print("Error: 'file' directory not found. Please create it and place necessary files (girl.bmp, buildHT.py, quantizedTable.py) inside.")
    exit()

# Check if helper files exist
if not os.path.exists('file/buildHT.py') or not os.path.exists('file/quantizedTable.py'):
    print("Error: buildHT.py or quantizedTable.py not found in the 'file' directory.")
    exit()
if not os.path.exists('file/girl.bmp'):
    print("Error: girl.bmp not found in the 'file' directory.")
    exit()

# Import helper modules from the 'file' directory
import sys
sys.path.append('file') # Add 'file' directory to Python path
try:
    import buildHT
    import quantizedTable
except ImportError as e:
    print(f"Error importing modules from 'file' directory: {e}")
    exit()

# --- Provided Functions (with minor adjustments/comments) ---

#1: RGB to YCbCr Conversion
def rgb2ycrcb(im_rgb):
    """Converts an RGB image (PIL Image object) to YCbCr format (NumPy array)."""
    # Ensure input is a NumPy array for easier processing
    if isinstance(im_rgb, Image.Image):
        width, height = im_rgb.size
        rgb_array = np.array(im_rgb).astype(float)
    else: # Assuming input is already a NumPy array HxWx3
        height, width, _ = im_rgb.shape
        rgb_array = im_rgb.astype(float)

    Y_Cb_Cr = np.empty((height, width, 3), dtype=float)

    # Transformation matrix (ITU-R BT.601 standard)
    # R = rgb_array[:, :, 0]
    # G = rgb_array[:, :, 1]
    # B = rgb_array[:, :, 2]

    # Y  =   0.299 * R + 0.587 * G + 0.114 * B
    # Cb = - 0.168736 * R - 0.331264 * G + 0.5 * B + 128
    # Cr =   0.5 * R - 0.418688 * G - 0.081312 * B + 128

    # Using the simplified coefficients provided in the original code
    # Note: These coefficients might slightly differ from standard JPEG,
    # and typically Cb/Cr are shifted by +128, which isn't done here yet.
    # The level shift (-128) is usually applied *before* DCT.
    Y_Cb_Cr[:, :, 0] = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
    Y_Cb_Cr[:, :, 1] = -0.169 * rgb_array[:, :, 0] - 0.331 * rgb_array[:, :, 1] + 0.5 * rgb_array[:, :, 2]
    Y_Cb_Cr[:, :, 2] = 0.5 * rgb_array[:, :, 0] - 0.419 * rgb_array[:, :, 1] - 0.081 * rgb_array[:, :, 2]

    return Y_Cb_Cr

# Note: The ycrcb2rgb function was provided but is not needed for compression.
# It would be needed for decompression.
# def ycrcb2rgb(bmp_image):
#     bmp_height, bmp_width, i = bmp_image.shape # Corrected order
#     rgb = np.empty((bmp_height, bmp_width, 3)) # Corrected order
#
#     for i_vertical in range(bmp_height):
#         for i_horizon in range(bmp_width):
#             ycrcb = bmp_image[i_vertical][i_horizon]
#             # Coefficients seem slightly off from standard inverse transformation
#             rgb[i_vertical][i_horizon][0] = (ycrcb * np.array([1.0,0,1.403])).sum() # R = Y + 1.403 * Cr
#             rgb[i_vertical][i_horizon][1] = (ycrcb * np.array([1.0, -0.344, -0.714])).sum() # G = Y - 0.344 * Cb - 0.714 * Cr
#             rgb[i_vertical][i_horizon][2] = (ycrcb * np.array([1.0, 1.773, 0])).sum() # B = Y + 1.773 * Cb
#
#     # Clamp values to [0, 255] and convert to uint8 for display/saving
#     rgb = np.clip(rgb, 0, 255).astype(np.uint8)
#     return rgb


#2: Adjust Size (Padding)
def adjustSize(nparray):
    """Pads a NumPy array (HxWxC) so height and width are multiples of 8."""
    inp_height, inp_width, i = nparray.shape # Corrected order
    pad_height = (8 - (inp_height % 8)) % 8 # More concise way to calculate padding
    pad_width = (8 - (inp_width % 8)) % 8
    if pad_height == 0 and pad_width == 0:
        return nparray # No padding needed
    # Padding applied to bottom and right edges
    nparray_padded = np.pad(nparray, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return nparray_padded

#4: Zigzag Scan
lable = [ # Standard Zigzag Scan Pattern Order
    [0,1,5,6,14,15,27,28],
    [2,4,7,13,16,26,29,42],
    [3,8,12,17,25,30,41,43],
    [9,11,18,24,31,40,44,53],
    [10,19,23,32,39,45,52,54],
    [20,22,33,38,46,51,55,60],
    [21,34,37,47,50,56,59,61],
    [35,36,48,49,57,58,62,63]
]

def bk2zip(mat):
    """Converts an 8x8 matrix to a 1D array using the Zigzag pattern."""
    result = np.empty(64)
    for i in range(8):
        for j in range(8):
            result[lable[i][j]] = mat[i][j]
    return result

# Note: zig2bk is for decompression
# def zig2bk(zig):
#     result = np.empty(shape=(8,8))
#     for i in range(8):
#         for j in range(8):
#             result[i][j] = zig[lable[i][j]]
#     return result

#5: Low Complement Binary Representation
def bit_need(n):
    """Calculates the minimum number of bits (category) needed to represent n."""
    n = int(n) # Ensure integer
    if n == 0:
        return 0
    magnitude = (int( np.ceil( np.log2(np.abs(n) + 1) ) ))
    return magnitude

def to_lcomp(n):
    """Converts an integer n to its low complement binary string representation."""
    n = int(n) # Ensure integer
    if n == 0:
        return "" # Category 0 has no additional bits
    else:
        bn = bit_need(n)
        if n > 0:
            # Positive numbers: standard binary representation
            return bin(n)[2:].zfill(bn) # zfill ensures correct length if leading zeros needed (shouldn't happen here)
        else:
            # Negative numbers: 1's complement of abs(n)
            # Calculate (2^bn - 1) - abs(n)
            # (2^bn - 1) is equivalent to a string of '1's of length bn
            # Use XOR for bitwise inversion relative to all 1s
            # Example: n=-3, bn=2. abs(n)=3 (0b11). Max val = 2^2-1 = 3 (0b11). 11 ^ 11 = 00.
            # Example: n=-5, bn=3. abs(n)=5 (0b101). Max val = 2^3-1 = 7 (0b111). 111 ^ 101 = 010.
            # The provided code `bin(int(bit_need(n)*"1",2)^np.abs(n))[2:].zfill(bit_need(n))` works correctly.
            return bin( ( (1 << bn) - 1 ) ^ np.abs(n) )[2:].zfill(bn)


# Note: from_lcomp is for decompression
# def from_lcomp(n_str):
#     # n = str(n) # Input should already be string
#     if n_str == "":
#         return 0
#     else:
#         if n_str[0] == "1": # Positive number
#             return int(n_str, 2)
#         else: # Negative number (starts with '0')
#             # Invert bits and make negative
#             # Equivalent to -( (2^len - 1) - int(n_str, 2) )
#             # Or using XOR: - ( int('1'*len(n_str), 2) ^ int(n_str, 2) )
#             length = len(n_str)
#             max_val = (1 << length) - 1
#             val = int(n_str, 2)
#             return - (max_val ^ val)


#6: Run-Length Encoding (RLE) for AC Coefficients
def runlenEn(zig):
    """Performs Run-Length Encoding on AC coefficients (zigzag array excluding DC)."""
    result = []
    ac_coeffs = zig[1:] # Exclude DC coefficient

    # Handle the case where all AC coefficients are zero
    if np.all(ac_coeffs == 0):
        return [(0, 0)] # EOB (End of Block)

    zero_count = 0
    for i in ac_coeffs:
        if i == 0:
            if zero_count == 15: # ZRL (Zero Run Length) marker
                result.append((15, 0))
                zero_count = 0
            else:
                zero_count += 1
        else:
            # Append (run_length, value)
            result.append((zero_count, int(i))) # Ensure value is int
            zero_count = 0 # Reset zero count

    # If the block ends with zeros, add EOB marker
    # The original code added EOB if zero_count != 0 at the end.
    # Standard JPEG adds EOB *unless* the last coefficient is non-zero
    # and fills exactly 63 AC terms. If the last run was (x, y),
    # and there are no more coefficients, we don't need EOB.
    # However, if the loop finished *because* we ran out of coefficients
    # *while* counting zeros, EOB is needed.
    # The original logic seems sufficient for this simplified case.
    # A strict JPEG encoder might handle this slightly differently.
    if zero_count > 0 or not result or result[-1] != (0,0): # Check if last element is already EOB
         # Ensure EOB is added if the block ends with zeros, unless the last element was ZRL (15,0)
         # If the last element was (15,0), we still need EOB if there were remaining zeros.
         # The simplest way is to always add EOB if the loop finishes.
         result.append((0, 0)) # EOB marker

    # Refinement: Remove redundant EOB if the last non-zero element filled the block
    # Find the index of the last non-zero element in the original AC coefficients
    last_nonzero_idx = -1
    for idx in range(len(ac_coeffs) - 1, -1, -1):
        if ac_coeffs[idx] != 0:
            last_nonzero_idx = idx
            break

    # Calculate total elements represented by RLE pairs (excluding EOB)
    elements_encoded = 0
    if result and result[-1] == (0,0): # Temporarily ignore potential EOB
        pairs_to_consider = result[:-1]
    else:
        pairs_to_consider = result

    for r, a in pairs_to_consider:
        elements_encoded += r + 1 # run + the non-zero value

    # If the last non-zero element was at index 62 (meaning all 63 AC coeffs are covered)
    # and the last RLE entry is EOB, remove the EOB.
    if last_nonzero_idx == 62 and elements_encoded == 63 and result and result[-1] == (0,0):
       result = result[:-1] # Remove EOB if block is perfectly full


    return result

# Note: runlenDe is for decompression
# def runlenDe(ra):
#     result = [] # Don't start with [0], AC coeffs don't include DC
#     for run, val in ra:
#         if run == 0 and val == 0: # EOB
#             break # Stop decoding for this block
#         result.extend([0] * run)
#         result.append(val)
#     # Pad with zeros to reach 63 AC coefficients if needed
#     while len(result) < 63:
#         result.append(0)
#     return np.array(result[:63]) # Return exactly 63 coeffs

#7: DC Huffman Coding
# Load Huffman Tables (assuming buildHT.py is in 'file' subdirectory)
try:
    dcLHT, acLHT, dcCHT, acCHT = buildHT.buildHT(buildHT.ht_default)
    # Note: Decoder tables (dcLHTd, etc.) are not needed for compression.
except Exception as e:
    print(f"Error loading Huffman tables using buildHT: {e}")
    exit()

def dcEn(dcHT, dc_diff):
    """Encodes a DC coefficient difference using the appropriate Huffman table."""
    dc_diff = int(dc_diff) # Ensure integer
    category = bit_need(dc_diff)
    lcomp_val = to_lcomp(dc_diff)
    try:
        huff_code = dcHT[category]
        result = huff_code + lcomp_val
        return result # Return only the bitstream string
    except KeyError:
        print(f"Error: Category {category} (from DC diff {dc_diff}) not found in DC Huffman table.")
        # Handle error appropriately, e.g., raise exception or return error indicator
        raise # Re-raise the exception for debugging

# Note: dcDe is for decompression
# def dcDe(dcHTd, dc_code): # Needs decoder table
#     # ... (implementation for decoding) ...

#8: AC Huffman Coding
def to_sk(r, a):
    """Combines run (r) and category (s=bit_need(a)) into a single byte for AC Huffman lookup."""
    s = bit_need(a)
    # JPEG standard uses Run/Size (Category) byte: RRRRSSSS
    return (r << 4) | s # Shift run left by 4 bits and OR with size

def acEn(acHT, run_ac):
    """Encodes AC RLE pairs using the appropriate Huffman table."""
    result = ""
    for r, a in run_ac:
        a = int(a) # Ensure value is integer
        if r == 0 and a == 0: # EOB marker (End of Block)
            key = 0x00 # Special code for EOB (0,0) -> (Run=0, Size=0)
        elif r == 15 and a == 0: # ZRL marker (Zero Run Length)
            key = 0xF0 # Special code for ZRL (15,0) -> (Run=15, Size=0)
        else:
            key = to_sk(r, a)

        lcomp_val = to_lcomp(a) # Value's amplitude bits
        try:
            huff_code = acHT[key]
            result += huff_code + lcomp_val
        except KeyError:
            print(f"Error: Key {key:#04x} (from RLE pair ({r},{a})) not found in AC Huffman table.")
            # Handle error
            raise # Re-raise the exception

    return result

# Note: acDe is for decompression
# def acDe(acHTd, ac_code): # Needs decoder table
#     # ... (implementation for decoding) ...

# --- Quantization Tables ---
q = 55 # Quality factor specified by user
try:
    lumQT, chrQT = quantizedTable.quantizedTable(q)
except Exception as e:
    print(f"Error loading quantization tables using quantizedTable: {e}")
    exit()

# Convert tables to NumPy arrays for element-wise operations
lumQT = np.array(lumQT).reshape(8, 8)
chrQT = np.array(chrQT).reshape(8, 8)

# --- Main Compression Function ---
def compress_image(image_path, lumQT, chrQT, dcLHT, acLHT, dcCHT, acCHT):
    """
    Performs the main JPEG-like compression steps.
    Returns the final concatenated bitstream as a string.
    """
    # 1. Load Image
    try:
        img_pil = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Input image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    print(f"Input image loaded: {img_pil.size} pixels, mode {img_pil.mode}")
    if img_pil.mode != 'RGB':
        print("Warning: Image mode is not RGB. Converting to RGB.")
        img_pil = img_pil.convert('RGB')

    # 2. Convert to YCbCr
    start_time = time.time()
    img_ycrcb = rgb2ycrcb(img_pil)
    print(f"Converted to YCbCr in {time.time() - start_time:.4f}s")

    # 3. Pad Image
    img_padded = adjustSize(img_ycrcb)
    padded_height, padded_width, _ = img_padded.shape
    print(f"Image padded to: ({padded_height}, {padded_width})")

    # 4. Initialize variables
    bitstream = ""
    prev_dc_Y = 0
    prev_dc_Cb = 0
    prev_dc_Cr = 0

    # 5. Process 8x8 Blocks
    print("Starting block processing (DCT, Quantization, Zigzag, RLE, Huffman)...")
    block_time_start = time.time()
    num_blocks = (padded_height // 8) * (padded_width // 8)
    processed_blocks = 0

    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            # Extract blocks for Y, Cb, Cr
            block_Y = img_padded[i:i+8, j:j+8, 0]
            block_Cb = img_padded[i:i+8, j:j+8, 1]
            block_Cr = img_padded[i:i+8, j:j+8, 2]

            # --- Process each component ---
            for block, quant_table, dc_table, ac_table, prev_dc_ref, comp_name in [
                (block_Y, lumQT, dcLHT, acLHT, 'prev_dc_Y', 'Y'),
                (block_Cb, chrQT, dcCHT, acCHT, 'prev_dc_Cb', 'Cb'),
                (block_Cr, chrQT, dcCHT, acCHT, 'prev_dc_Cr', 'Cr')
            ]:
                # a. Level Shift (Center data around 0)
                block_shifted = block - 128.0

                # b. Apply 2D DCT
                dct_block = dctn(block_shifted, type=2, norm='ortho') # Type II DCT is standard

                # c. Quantize
                quantized_block = np.round(dct_block / quant_table).astype(int)

                # d. Zigzag Scan
                zigzag_array = bk2zip(quantized_block)

                # e. Separate DC and AC coefficients
                dc_coeff = int(zigzag_array[0]) # Ensure DC is integer
                ac_coeffs = zigzag_array # Keep AC as part of the array for RLE

                # f. DC Coefficient Processing (Differential Coding + Huffman)
                # Need to update the *actual* previous DC variable, not just pass its value
                if comp_name == 'Y':
                    dc_diff = dc_coeff - prev_dc_Y
                    prev_dc_Y = dc_coeff
                elif comp_name == 'Cb':
                    dc_diff = dc_coeff - prev_dc_Cb
                    prev_dc_Cb = dc_coeff
                else: # Cr
                    dc_diff = dc_coeff - prev_dc_Cr
                    prev_dc_Cr = dc_coeff

                dc_bits = dcEn(dc_table, dc_diff)
                bitstream += dc_bits

                # g. AC Coefficient Processing (RLE + Huffman)
                rle_pairs = runlenEn(ac_coeffs) # Pass the whole zigzag array
                ac_bits = acEn(ac_table, rle_pairs)
                bitstream += ac_bits

            processed_blocks += 1
            if processed_blocks % 100 == 0: # Print progress update
                 print(f"  Processed {processed_blocks}/{num_blocks} blocks...")

    print(f"Block processing completed in {time.time() - block_time_start:.4f}s")
    print(f"Total bitstream length: {len(bitstream)} bits")

    return bitstream

# --- Execution ---
input_image_path = 'file/test16.bmp'
print(f"Starting compression for '{input_image_path}' with Q={q}...")

final_bitstream = compress_image(input_image_path, lumQT, chrQT, dcLHT, acLHT, dcCHT, acCHT)

if final_bitstream is not None:
    print(final_bitstream)
