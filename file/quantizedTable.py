import numpy as np
def quantizedTable(quality=50):
    std_lumQT = np.array(          # standard luminance quantized table
        [[ 16,  11,  10,  16,  24,  40,  51,  61],
         [ 12,  12,  14,  19,  26,  58,  60,  55],
         [ 14,  13,  16,  24,  40,  57,  69,  56],
         [ 14,  17,  22,  29,  51,  87,  80,  62],
         [ 18,  22,  37,  56,  68, 109, 103,  77],
         [ 24,  35,  55,  64,  81, 104, 113,  92],
         [ 49,  64,  78,  87, 103, 121, 120, 101],
         [ 72,  92,  95,  98, 112, 100, 103,  99]])

    std_chrQT = np.array(        # standard chrominance quantized table
        [[ 17,  18,  24,  47,  99,  99,  99,  99],
         [ 18,  21,  26,  66,  99,  99,  99,  99],
         [ 24,  26,  56,  99,  99,  99,  99,  99],
         [ 47,  66,  99,  99,  99,  99,  99,  99],
         [ 99,  99,  99,  99,  99,  99,  99,  99],
         [ 99,  99,  99,  99,  99,  99,  99,  99],
         [ 99,  99,  99,  99,  99,  99,  99,  99],
         [ 99,  99,  99,  99,  99,  99,  99,  99]])
    
    qualityScale = 5000/quality if(quality < 50) else 200-quality*2
    lumQT = np.floor((std_lumQT*qualityScale+50)/100).clip(1,255).astype(int)
    chrQT = np.floor((std_chrQT*qualityScale+50)/100).clip(1,255).astype(int)
    return lumQT,chrQT