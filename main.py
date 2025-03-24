from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dctn

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


def cut2block(Y_Cb_Cr):

    inp_width, inp_height = inp_image.size
    if inp_height % 8 != 0 or inp_width % 8 != 0:
        #todo: padding
        pass

    Yi= np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))
    Cb= np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))
    Cr= np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))

    for i_raw in range(0,int(inp_width/8)):
        for i_col in range(0,int(inp_height/8)):
            Yi[i_raw*int(inp_height/8) + i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,0])
            Cb[i_raw*int(inp_height/8) + i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,1])
            Cr[i_raw*int(inp_height/8) + i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,2])
            #print(Y_Cb_Cr_r[i_raw*int(inp_height/8) + i_col][0])
            
    return Yi,Cb,Cr



# main
inp_image = Image.open( 'file/test16.bmp' )

# turn to ycryb
ycrcb = rgb2ycrcb(inp_image)
# print(ycrcb[:8,:8,1 ].round().astype(int))

# print("----------------")
# spilt to 8x8 block
Y_channel_8x8,Cb_channel_8x8,Cr_channel_8x8 = cut2block(ycrcb)
#print(cut2block(ycrcb)[0].round().astype(int))

# turn Y channel to -128~128
Y_channel_8x8 -= 128

# DCT
y_dcted = []
Cr_dcted = []
Cb_dcted = []

for yc,cbc,crc in zip(Y_channel_8x8,Cb_channel_8x8,Cr_channel_8x8):
    y_dcted.append(dctn(yc, norm='ortho',axes=(0,1)))
    Cr_dcted.append(dctn(crc, norm='ortho',axes=(0,1)))
    Cb_dcted.append(dctn(cbc, norm='ortho',axes=(0,1)))

#print(Cr_dcted[3].round().astype(int))
    
# Quantization
from file.quantizedTable import quantizedTable
lumQT,chrQT = quantizedTable(55)

qy = []
qcb = []
qcr = []

for i in range(len(y_dcted)):
    qy.append((y_dcted[i]/lumQT).round().astype(int))
    qcb.append((Cb_dcted[i]/chrQT).round().astype(int))
    qcr.append((Cr_dcted[i]/chrQT).round().astype(int))

# jpg encode
    
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

def zigzag(block):
    result = np.empty(64)
    for i in range(8):
        for j in range(8):
            result[lable[i][j]] = block[i][j]
    return result

y64,cb64,cr64 = [],[],[]

for i in range(len(qy)):
    y64.append(zigzag(qy[i]))
    cb64.append(zigzag(qcb[i]))
    cr64.append(zigzag(qcr[i]))

# DPCM

y64_DC = []
cb64_DC = []
cr64_DC = []

for i in range(len(y64)):
    y64_DC.append(y64[i][0])
    cb64_DC.append(cb64[i][0])
    cr64_DC.append(cr64[i][0])

def DCPM(list):

    result = []
    result.append(list[0])
    for i in range(1,len(list)):
        result.append(list[i] - list[i-1])
    
    return result

y_DC = DCPM(y64_DC)
cb_DC = DCPM(cb64_DC)
cr_DC = DCPM(cr64_DC)   

# RLC AC
def RLC(list):
    result = []
    zero_count = 0
    for i in list[1:]:
        if i == 0:
            if zero_count == 15:
                zero_count = 0
                result.append((15,0))
                continue
            zero_count += 1
            continue
        result.append((zero_count,int(i)))
        zero_count = 0
    # todo case All zero and last not zero
    result.append((0,0))
    return result

# print(y64[0])
y_ac_RLC = []
cb_ac_RLC = []
cr_ac_RLC = []
for i in y64:
    y_ac_RLC.append(RLC(i))
for i in cb64:
    cb_ac_RLC.append(RLC(i))
for i in cr64:
    cr_ac_RLC.append(RLC(i))

# 1 complement

def bit_need(n):
    magnitude = (int( np.ceil( np.log2(np.abs(n) + 1) ) ))
    return (magnitude)
        

def one_complement(n):
    if n == 0:
        return ""
    else:
        return bin(n)[2:] if n > 0 else bin(int(bit_need(n)*"1",2)^np.abs(n))[2:].zfill(bit_need(n))


# for i in range(0,10):
#     print(str(i)+":"+str(one_complement(i)))

# for i in range(0,-10,-1):
#     print(str(i)+":"+str(one_complement(i)))



# huffman
# DC
import file.buildHT as buildHT

y_DC_huff = []
cb_DC_huff = []
cr_DC_huff = []

def huffman(n,i):
    result = (buildHT.buildHT(buildHT.ht_default)[i][bit_need(n)])+one_complement(int(n))
    return result


for i in y_DC:
    y_DC_huff.append(huffman(i,0))

for l,lc in zip([cb_DC,cr_DC],[cb_DC_huff,cr_DC_huff]):
    for i in l:
        lc.append(huffman(i,2))

#AC

def to_sk(r,a):
    return r * 16 + bit_need(a)
def huffman_AC(n,i):
    result = (buildHT.buildHT(buildHT.ht_default)[i][to_sk(n[0],n[1])])+one_complement(int(n[1]))
    return result

y_aC_huff = []
cb_aC_huff = []
cr_aC_huff = []

for i in y_ac_RLC:
    l = []
    for j in i:
        l.append(huffman_AC(j,1))
    y_aC_huff.append(l)

for i in cb_ac_RLC:
    l = []
    for j in i:
        l.append(huffman_AC(j,3))
    cb_aC_huff.append(l)

for i in cr_ac_RLC:
    l = []
    for j in i:
        l.append(huffman_AC(j,3))
    cr_aC_huff.append(l)


# encode

final_result = ""

for i in range(len(y_DC_huff)):

    final_result += y_DC_huff[i]
    for j in y_aC_huff[i]:
        final_result += j
    final_result += cb_DC_huff[i]
    for j in cb_aC_huff[i]:
        final_result += j
    final_result += cr_DC_huff[i]
    for j in cr_aC_huff[i]:
        final_result += j
print(final_result)



