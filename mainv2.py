from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn
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
    bmp_width, bmp_height,i = bmp_image.shape
    rgb = np.empty((bmp_width,bmp_height, 3))

    for i_vertical in range(bmp_height):
        for i_horizon in range(bmp_width):
            
            ycrcb = bmp_image[i_vertical][i_horizon]
            rgb[i_vertical][i_horizon][0] = (ycrcb * np.array([1.0,0,1.403])).sum()
            rgb[i_vertical][i_horizon][1] = (ycrcb * np.array([1.0, -0.344, -0.714])).sum()
            rgb[i_vertical][i_horizon][2] = (ycrcb * np.array([1.0, 1.773, 0])).sum()
    
    return rgb

#2
def adjustSize(nparray):
    inp_width, inp_height, i = nparray.shape
    if inp_height % 8 != 0 or inp_width % 8 != 0:
        # Padding
        pad_height = 8 - (inp_height % 8) if inp_height % 8 != 0 else 0
        pad_width = 8 - (inp_width % 8) if inp_width % 8 != 0 else 0
        nparray = np.pad(nparray, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    return nparray

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
    if zero_count != 0:
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
# print(runlenDe(runlenEn(zig)))


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
    result = ""
    for r,a in run_ac:
        result += acHT[to_sk(r,a)]+to_lcomp(a)
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
from file.quantizedTable import quantizedTable
lumQT,chrQT = quantizedTable(55)


inp_image = Image.open( 'file/girl.bmp' )




# 
