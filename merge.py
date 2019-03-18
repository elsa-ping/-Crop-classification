import numpy as np
import tifffile as tiff

f_image = np.zeros((17810,50362,8),dtype=np.uint16)  # 创建数组    17810 50362 8


for i in range(1,7,1):

    img = tiff.imread('./cut images_raw/new_image_%d.tif'%i)
    img = img.transpose([1,2,0])

    height,width,channel = img.shape
    # print(images[i-1].shape)
    
    col_start = (i-1)*10000
    col_end = col_start + width 
    f_image[:,col_start:col_end,:] = img

# print(f_image.shape)
tiff.imsave('./merge_image.tif',f_image)
# f_image = np.ones((17810,50362,8))
