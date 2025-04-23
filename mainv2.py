from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn, idctn
import time

#1
def rgb2ycrcb(bmp_image):
    Y_Cb_Cr = np.empty((bmp_image.height, bmp_image.width, 3))

    for i_vertical in range(bmp_image.height):
        for i_horizon in range(bmp_image.width):

            rgb = bmp_image.getpixel((i_horizon, i_vertical))
            Y_Cb_Cr[i_vertical][i_horizon][0] = (rgb * np.array([0.299, 0.587, 0.114])).sum()
            Y_Cb_Cr[i_vertical][i_horizon][1] = (rgb * np.array([-0.169, -0.331, 0.5])).sum()
            Y_Cb_Cr[i_vertical][i_horizon][2] = (rgb * np.array([0.5, -0.419, -0.081])).sum()

    bmp_image.close()
    return Y_Cb_Cr

def ycrcb2rgb(bmp_image):
    height, width, _ = bmp_image.shape
    rgb = np.empty((height, width, 3), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            y, cb, cr = bmp_image[i, j]
            r = y + 1.403 * cr
            g = y - 0.344 * cb - 0.714 * cr
            b = y + 1.773 * cb
            rgb[i, j] = [r, g, b]

    return np.clip(rgb, 0, 255).astype(np.uint8)

#2
def adjustSize(nparray):
    inp_width, inp_height, i = nparray.shape
    if inp_height % 8 != 0 or inp_width % 8 != 0:
        # Padding
        pad_height = 8 - (inp_height % 8) if inp_height % 8 != 0 else 0
        pad_width = 8 - (inp_width % 8) if inp_width % 8 != 0 else 0
        nparray = np.pad(nparray, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return nparray

def cut2block(Y_Cb_Cr):
    inp_height, inp_width, _ = Y_Cb_Cr.shape
    print("Original shape:", Y_Cb_Cr.shape)

    if inp_height % 8 != 0 or inp_width % 8 != 0:
        pad_height = (8 - (inp_height % 8)) if inp_height % 8 != 0 else 0
        pad_width = (8 - (inp_width % 8)) if inp_width % 8 != 0 else 0
        Y_Cb_Cr = np.pad(Y_Cb_Cr, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        inp_height, inp_width, _ = Y_Cb_Cr.shape

    print("Padded shape:", Y_Cb_Cr.shape)

    blocks_per_row = inp_width // 8
    blocks_per_col = inp_height // 8
    total_blocks = blocks_per_col * blocks_per_row

    Yi = np.zeros((total_blocks, 8, 8))
    Cb = np.zeros((total_blocks, 8, 8))
    Cr = np.zeros((total_blocks, 8, 8))

    for i_row in range(blocks_per_col):
        for i_col in range(blocks_per_row):
            idx = i_row * blocks_per_row + i_col
            Yi[idx] = Y_Cb_Cr[i_row*8:(i_row+1)*8, i_col*8:(i_col+1)*8, 0]
            Cb[idx] = Y_Cb_Cr[i_row*8:(i_row+1)*8, i_col*8:(i_col+1)*8, 1]
            Cr[idx] = Y_Cb_Cr[i_row*8:(i_row+1)*8, i_col*8:(i_col+1)*8, 2]

    return Yi, Cb, Cr


#4
lable = [
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
    result = np.empty(64)
    for i in range(8):
        for j in range(8):
            result[lable[i][j]] = mat[i][j]
    return result

def zig2bk(zig):
    result = np.empty(shape=(8,8))
    for i in range(8):
        for j in range(8):
            result[i][j] = zig[lable[i][j]]
    return result

#5
def bit_need(n):
    magnitude = (int( np.ceil( np.log2(np.abs(n) + 1) ) ))
    return (magnitude)
        
def to_lcomp(n):
    n = int(n)
    if n == 0:
        return ""
    else:
        return bin(n)[2:] if n > 0 else bin(int(bit_need(n)*"1",2)^np.abs(n))[2:].zfill(bit_need(n))

def from_lcomp(n):
    n = str(n)
    if n == "":
        return 0
    else:
        if n[0] == "1":
            return int(n,2)
        else:
            return int(int(n,2) ^ int("1"*len(n),2))*-1

# print(to_lcomp(-123),to_lcomp(0),to_lcomp(-1),to_lcomp(2))
# print(from_lcomp("0000100"),from_lcomp(""),from_lcomp(0),from_lcomp(10))

#6
def runlenEn(zig):
    result = []
    # all zero case
    if zig[1:].sum() == 0:
        return [(0,0)]

    zero_count = 0
    for i in zig[1:]:
        if i == 0:
            if zero_count == 15:
                zero_count = 0
                result.append((15,0))
                continue
            zero_count += 1
            continue
        result.append((zero_count,int(i)))
        zero_count = 0
    while True:
        if result[-1] == (15,0):
            result.pop()
        else:
            break
    result.append((0,0))
    return result

def runlenDe(ra):
    result = [0]
    for i in ra:
        if i[0] == 15:
            result.extend([0]*15)
        else:
            result.extend([0]*i[0])
            result.append(i[1])
    return np.array(result)

# zig=np.array([9,8,9,3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
# print(runlenEn(zig))
#print(runlenDe([(0, 26), (0, 4), (0, 6), (0, -13), (0, -12), (0, -1), (0, 3), (0, -7), (0, -2), (0, 1), (0, 3), (0, 2), (1, 3), (0, -1), (0, -2), (0, 1), (0, 2), (0, 1), (0, -2), (1, 1), (1, -1), (0, -1), (0, 1), (1, 1), (2, 1), (0, 1), (10, -1), (0, 0)]))


#7
import file.buildHT as buildHT
dcLHT,acLHT,dcCHT,acCHT = buildHT.buildHT(buildHT.ht_default)
dcLHTd,acLHTd,dcCHTd,acCHTd = buildHT.buildHT(buildHT.ht_default,'decode')

def dcEn(dcHT,dc_diff):
    result = dcHT[bit_need(dc_diff)]+to_lcomp(dc_diff)
    return result,len(result)

def dcDe(dcHT, dc_code):
    for i in range(1, len(dc_code) + 1):
        prefix = dc_code[:i]
        if prefix in dcHT:
            category = dcHT[prefix]
            lcomp_bits = dc_code[i:i + category]
            return from_lcomp(lcomp_bits), i + category


# print(dcEn(dcLHT, 21),dcEn(dcCHT, -92))
# print(dcDe(dcLHTd,'11010101'),dcDe(dcCHTd,'11111100100011'))

#8
def to_sk(r,a):
    return r * 16 + bit_need(a)

def acEn(acHT, run_ac):
    # print(run_ac)
    # print("AC:")
    result = ""
    for r,a in run_ac:
        result += acHT[to_sk(r,a)]+to_lcomp(a)
        #print(acHT[to_sk(r,a)]+to_lcomp(a))
    return result

def acDe(acHT, ac_code):
    result = []
    i = 0
    while i < len(ac_code):
        found = False
        for j in range(1, 17):
            prefix = ac_code[i:i + j]
            if prefix in acHT:
                val = acHT[prefix]
                r = val >> 4
                a = val & 0xF
                lcomp_bits = ac_code[i + j:i + j + a]
                result.append((r, from_lcomp(lcomp_bits)))
                i += j + a
                found = True
                break
    return result


# print(acEn(acLHT, [(0,9),(7,12),(9,4),(0,15),(3,1),(0,0)]))
# print(acDe(acLHTd,'101110011111111110101111110011111111101111111001011111111101011010'))

inp_image = []
def encode(file_name = 'file/test16.bmp'):
    global inp_image
    inp_image = Image.open(file_name)
    # main
    from file.quantizedTable import quantizedTable
    lumQT,chrQT = quantizedTable(55)

    start_time = time.time()
    ycrycb = rgb2ycrcb(inp_image)
    ycrcb_resize = adjustSize(ycrycb)
    block_yc,block_crc,block_cbc = cut2block(ycrcb_resize)
    block_yc -= 128

    y_dcted = []
    Cr_dcted = []
    Cb_dcted = []
    for yc,cbc,crc in zip(block_yc,block_cbc,block_crc):
        y_dcted.append(dctn(yc, norm='ortho',axes=(0,1)))
        Cr_dcted.append(dctn(crc, norm='ortho',axes=(0,1)))
        Cb_dcted.append(dctn(cbc, norm='ortho',axes=(0,1)))

    q_y = []
    q_cb = []
    q_cr = []

    for i in range(len(y_dcted)):
        q_y.append((y_dcted[i]/lumQT).round().astype(int))
        q_cb.append((Cb_dcted[i]/chrQT).round().astype(int))
        q_cr.append((Cr_dcted[i]/chrQT).round().astype(int))

    zip_y = []
    zip_cb = []
    zip_cr = []

    for y,cy,cb in zip(q_y,q_cr,q_cb):
        zip_y.append(bk2zip(y))
        zip_cb.append(bk2zip(cy))
        zip_cr.append(bk2zip(cb))
        

    Dc_y = [(i[0]) for i in zip_y]
    Dc_cb = [i[0] for i in zip_cb]
    Dc_cr = [i[0] for i in zip_cr]

    DPCM_y = []
    DPCM_cb = []
    DPCM_cr = []

    for i in range(len(Dc_y)):
        if i == 0:
            DPCM_y.append(Dc_y[i])
            DPCM_cb.append(Dc_cb[i])
            DPCM_cr.append(Dc_cr[i])
        else:
            DPCM_y.append(Dc_y[i] - Dc_y[i-1])
            DPCM_cb.append(Dc_cb[i] - Dc_cb[i-1])
            DPCM_cr.append(Dc_cr[i] - Dc_cr[i-1])


    AC_run_y = []
    AC_run_cb = []
    AC_run_cr = []
    for i in range(len(zip_y)):
        AC_run_y.append(runlenEn(zip_y[i]))
        AC_run_cb.append(runlenEn(zip_cb[i]))
        AC_run_cr.append(runlenEn(zip_cr[i]))

    print("Y ac: ",AC_run_y[1])

    DC_huff_y = []
    DC_huff_cb = []
    DC_huff_cr = []
    for i in range(len(DPCM_y)):
        DC_huff_y.append(dcEn(dcLHT,DPCM_y[i]))
        DC_huff_cb.append(dcEn(dcCHT,DPCM_cb[i]))
        DC_huff_cr.append(dcEn(dcCHT,DPCM_cr[i]))

    AC_huff_y = []
    AC_huff_cb = []
    AC_huff_cr = []
    for i in range(len(AC_run_y)):
        AC_huff_y.append(acEn(acLHT,AC_run_y[i]))
        AC_huff_cb.append(acEn(acCHT,AC_run_cb[i]))
        AC_huff_cr.append(acEn(acCHT,AC_run_cr[i]))

    result = ""

    for i in range(len(DC_huff_cb)):
        result += DC_huff_y[i][0] + AC_huff_y[i] + DC_huff_cb[i][0] + AC_huff_cb[i] + DC_huff_cr[i][0] + AC_huff_cr[i]

    #print(result)
    print("Time: ",time.time() - start_time)

    return result


def decode(code,w,h):
    
    # main
    from file.quantizedTable import quantizedTable
    lumQT,chrQT = quantizedTable(55)

    import file.buildHT as buildHT
    dcLHTe,acLHTe,dcCHTe,acCHTe = buildHT.buildHT(buildHT.ht_default,'encode')

    codes = code

    runt = []

    y_dc = []
    cb_dc = []
    cr_dc = []

    y_ac = []
    cb_ac = []
    cr_ac = []

    c = 0
    while codes != "":


        for i in dcLHTe:
            #print(codes[:len(dcLHTe[i])],dcLHTe[i],codes[:len(dcLHTe[i])] == dcLHTe[i])
            if codes[:len(dcLHTe[i])] == dcLHTe[i]:
                y_dc.append(from_lcomp(codes[len(dcLHTe[i]):len(dcLHTe[i])+i]))
                codes = codes[len(dcLHTe[i])+i:]
                break
        runts = []


        while True:
            for i in acLHTe:
                if codes[:len(acLHTe[i])] == acLHTe[i]:
                    r = (i >> 4)
                    a = (i & 0xF)
                    lcomp_bits = codes[len(acLHTe[i]):len(acLHTe[i])+a]
                    runts.append((r,from_lcomp(lcomp_bits)))

                    codes = codes[len(acLHTe[i])+a:]

                    break
            
            if runts[-1] == (0,0):
                break
        y_ac.append(runts)


 

        for i in dcCHTe:

            if codes[:len(dcCHTe[i])] == dcCHTe[i]:
                cb_dc.append(from_lcomp(codes[len(dcCHTe[i]):len(dcCHTe[i])+i]))
                codes = codes[len(dcCHTe[i])+i:]
                break
        runts = []
        while True:
            for i in acCHTe:
                if codes[:len(acCHTe[i])] == acCHTe[i]:
                    r = (i >> 4)
                    a = (i & 0xF)
                    lcomp_bits = codes[len(acCHTe[i]):len(acCHTe[i])+a]
                    runts.append((r,from_lcomp(lcomp_bits)))
                    codes = codes[len(acCHTe[i])+a:]
                    break
            if runts[-1] == (0,0):
                break
        cb_ac.append(runts)

        for i in dcCHTe:

            if codes[:len(dcCHTe[i])] == dcCHTe[i]:
                cr_dc.append(from_lcomp(codes[len(dcCHTe[i]):len(dcCHTe[i])+i]))
                codes = codes[len(dcCHTe[i])+i:]
                break
        runts = []
        while True:
            for i in acCHTe:
                if codes[:len(acCHTe[i])] == acCHTe[i]:
                    r = (i >> 4)
                    a = (i & 0xF)
                    lcomp_bits = codes[len(acCHTe[i]):len(acCHTe[i])+a]
                    runts.append((r,from_lcomp(lcomp_bits)))
                    codes = codes[len(acCHTe[i])+a:]
                    break
            if runts[-1] == (0,0):
                break
        cr_ac.append(runts)


    new_y_dc = []
    new_cb_dc = []
    new_cr_dc = []
    for i in range(len(y_dc)):
        new_y_dc.append(sum(y_dc[:i+1]))
        new_cb_dc.append(sum(cb_dc[:i+1]))
        new_cr_dc.append(sum(cr_dc[:i+1]))


    block_y = []
    block_cb = []
    block_cr = []

    for i in range(len(new_y_dc)):
        zigzag = runlenDe(y_ac[i])
        zigzag[0] = new_y_dc[i]
        zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), mode='constant', constant_values=0)
        block_y.append(zig2bk(np.array(zigzag)))

    for i in range(len(new_cb_dc)):
        zigzag = runlenDe(cb_ac[i])
        zigzag[0] = new_cb_dc[i]
        zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), mode='constant', constant_values=0)
        block_cb.append(zig2bk(np.array(zigzag)))
    
    for i in range(len(new_cr_dc)):
        zigzag = runlenDe(cr_ac[i])
        zigzag[0] = new_cr_dc[i]
        zigzag = np.pad(zigzag, (0, 64 - len(zigzag)), mode='constant', constant_values=0)
        block_cr.append(zig2bk(np.array(zigzag)))

    deqt_y = []
    deqt_cb = []
    deqt_cr = []

    for i in range(len(block_y)):
        deqt_y.append(block_y[i] * lumQT)
        deqt_cb.append(block_cb[i] * chrQT)
        deqt_cr.append(block_cr[i] * chrQT)

    #dct
    y_idcted = []
    cb_idcted = []
    cr_idcted = []

    for i in range(len(deqt_y)):
        y_idcted.append(idctn(deqt_y[i], norm='ortho', axes=(0, 1)))
        cb_idcted.append(idctn(deqt_cb[i], norm='ortho', axes=(0, 1)))
        cr_idcted.append(idctn(deqt_cr[i], norm='ortho', axes=(0, 1)))
    
    y_idcted = np.array(y_idcted)
    cb_idcted = np.array(cb_idcted)
    cr_idcted = np.array(cr_idcted)

    y_idcted = y_idcted + 128

    
    padded_height = ((h + 7) // 8) * 8 
    padded_width = ((w + 7) // 8) * 8 

    ycrcb_combined = np.zeros((padded_height, padded_width, 3))


    blocks_per_row = padded_width // 8
    blocks_per_col = padded_height // 8

    block_index = 0
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            ycrcb_combined[i*8:(i+1)*8, j*8:(j+1)*8, 0] = y_idcted[block_index]
            ycrcb_combined[i*8:(i+1)*8, j*8:(j+1)*8, 1] = cb_idcted[block_index]
            ycrcb_combined[i*8:(i+1)*8, j*8:(j+1)*8, 2] = cr_idcted[block_index]
            block_index += 1

    print("Padded shape:", ycrcb_combined.shape)
    rgb_image = ycrcb2rgb(ycrcb_combined[:h, :w, :])
    print("RGB shape:", rgb_image.shape)
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(rgb_image, 'RGB')
    result_image.save('decoded_image.bmp')


#decode("11010100110101101010010010011010110010101100110000111100000010100101110110110111100001010010110001010111001110000000011100111100100111111101001010010101101011000011001000011101010110100100011001001011011010110101100101010100100000101111110101000001110001110000011110001110101101010001001001010111010110101001101101000001011011101000110101001001001011101001010001101001100101110010010010100010110011110001011011001110010101000111000001111001111000111111110011110001010101110000101110111001010110010001100100110100100100101101010010100111001101001001000101001011101001110100011000011110000001011101111000100001000011000001010011100000111011000010000001100000100000000110101001100111101110010101000101111010011010000")
img = Image.open('file\坤輿萬國全圖.jpg')
decode(encode("file\坤輿萬國全圖.jpg"),img.width,img.height)