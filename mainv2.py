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

def cut2block(Y_Cb_Cr):


    inp_width, inp_height = inp_image.size
    #print(Y_Cb_Cr.shape)
    if inp_height % 8 != 0 or inp_width % 8 != 0:
        # Padding
        pad_height = 8 - (inp_height % 8) if inp_height % 8 != 0 else 0
        pad_width = 8 - (inp_width % 8) if inp_width % 8 != 0 else 0
        Y_Cb_Cr = np.pad(Y_Cb_Cr, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

    Yi = np.zeros((round(inp_width / 8) * round(inp_height / 8), 8, 8))
    Cb = np.zeros((round(inp_width / 8) * round(inp_height / 8), 8, 8))
    Cr = np.zeros((round(inp_width / 8) * round(inp_height / 8), 8, 8))

    for i_raw in range(0, round(inp_width / 8)):
        for i_col in range(0, round(inp_height / 8)):
            try:
                Yi[i_raw * round(inp_height / 8) + i_col] = (Y_Cb_Cr[i_raw * 8:i_raw * 8 + 8, i_col * 8:i_col * 8 + 8, 0])
                Cb[i_raw * round(inp_height / 8) + i_col] = (Y_Cb_Cr[i_raw * 8:i_raw * 8 + 8, i_col * 8:i_col * 8 + 8, 1])
                Cr[i_raw * round(inp_height / 8) + i_col] = (Y_Cb_Cr[i_raw * 8:i_raw * 8 + 8, i_col * 8:i_col * 8 + 8, 2])
            except Exception as e:
                print("Error:", e)
                print(Yi[i_raw * round(inp_height / 8) + i_col])
                print(Y_Cb_Cr[i_raw * 8:i_raw * 8 + 8, i_col * 8:i_col * 8 + 8, 0])
                time.sleep(10)


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
    if zero_count != 0:
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



# main
from file.quantizedTable import quantizedTable
lumQT,chrQT = quantizedTable(55)

start_time = time.time()
inp_image = Image.open( 'file/girl.bmp' )
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

print(result)
print("Time: ",time.time() - start_time)