from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# def rgb2ycrcb(bmp_image): #old
#     Y_Cb_Cr = np.empty((bmp_image.height, bmp_image.width, 3))

#     for i_vertical in range(bmp_image.height):
#         for i_horizon in range(bmp_image.width):

#             rgb = bmp_image.getpixel((i_horizon, i_vertical))
#             Y_Cb_Cr[i_vertical][i_horizon][0] = (rgb * np.array([0.299, 0.587, 0.114])).sum()
#             Y_Cb_Cr[i_vertical][i_horizon][1] = (rgb * np.array([-0.169, -0.331, 0.5])).sum()
#             Y_Cb_Cr[i_vertical][i_horizon][2] = (rgb * np.array([0.5, -0.419, -0.081])).sum()

#     return Y_Cb_Cr

def rgb2ycrcb(bmp_image):

    Y_Cb_Cr = np.empty((bmp_image.height, bmp_image.width, 3))
    core = np.array([[[0.299, 0.587, 0.114]]*bmp_image.width]*bmp_image.height)
    Y_Cb_Cr[:,:,0] = (bmp_image * core).sum(axis=2)
    core = np.array([[[-0.169, -0.331, 0.5]]*bmp_image.width]*bmp_image.height,)
    Y_Cb_Cr[:,:,1] = (bmp_image * core).sum(axis=2)
    core = np.array([[[0.5, -0.419, -0.081]]*bmp_image.width]*bmp_image.height,)
    Y_Cb_Cr[:,:,2] = (bmp_image * core).sum(axis=2)

    return Y_Cb_Cr


def ycrcb2rgb(bmp_image):

    rgb = np.empty((bmp_image.height, bmp_image.width, 3))
    core = np.array([[[1.0,0,1.403]]*bmp_image.width]*bmp_image.height)
    rgb[:,:,0] = (bmp_image * core).sum(axis=2)
    core = np.array([[[1.0, -0.344, -0.714]]*bmp_image.width]*bmp_image.height,)
    rgb[:,:,1] = (bmp_image * core).sum(axis=2)
    core = np.array([[[1.0, -0.344, -0.714]]*bmp_image.width]*bmp_image.height,)
    rgb[:,:,2] = (bmp_image * core).sum(axis=2)

    return rgb


def show_channel(Y_Cb_Cr):
    ycrcbs = ycrcb[:,:,0],ycrcb[:,:,1],ycrcb[:,:,2],ycrcb[:,:,2]
    fig,axs = plt.subplots(2,2,figsize=(10, 10))  # 設定圖表大小

    for n,ax in enumerate(axs.flat):
        data = ycrcbs[n]

        ax.imshow(data, cmap='gray')
        ax.grid(True, which='both',color="black",linewidth=1)

        ax.set_xticks(np.arange(-.5, data.shape[1], 1))
        ax.set_yticks(np.arange(-.5, data.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks(np.arange(-.5, data.shape[1], 1))
        ax.set_yticks(np.arange(-.5, data.shape[0], 1))
        ax.set_xticklabels([int(x) for x in range(data.shape[1])], minor=True)
        ax.set_yticklabels([int(y) for y in range(data.shape[0])], minor=True)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, int(data[i, j]), ha='center', va='center', color='black', fontsize=7)

        ax.axis('equal')

    plt.show()



def cut2block(Y_Cb_Cr):
    # cut to 8x8
    inp_width, inp_height = inp_image.size
    if inp_height % 8 != 0 or inp_width % 8 != 0:
        # 
        pass


    Y_channel = np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))
    Cb_channel = np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))
    Cr_channel = np.empty((int(inp_width/8) * int(inp_height/8), 8, 8))

    for i_raw in range(0,int(inp_width/8)):
        for i_col in range(0,int(inp_height/8)):
            Y_channel[i_raw * i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,0])
            Cb_channel[i_raw * i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,1])
            Cr_channel[i_raw * i_col] = (Y_Cb_Cr[i_raw*8:i_raw*8+8,i_col*8:i_col*8+8,2])
    return Y_channel, Cb_channel, Cr_channel

# start
inp_image = Image.open( 'drive-download-20250305T052634Z-001/test16.bmp' )
inp_bmap = np.array(inp_image)

start_time = time.time()
ycrcb = rgb2ycrcb(inp_image)
print("--- %s seconds ---" % (time.time() - start_time))
inp_image.close()
show_channel(ycrcb)

print(cut2block(ycrcb)[0][3])

