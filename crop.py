import tifffile as tiff
import numpy as np


img = tiff.imread('./image.tif')

# img = img.transpose([1, 2, 0])


col = 0
n = 0
while(col + 10000 < 50362):
    tiff.imsave('./image_%d.tif'%n, img[:, :, n*10000:(n+1)*10000])
    col = col + 10000
    n = n+1

tiff.imsave('./image_%d.tif'%n, img[:, :, -10000:])

# tiff.imsave('./image_1.tif', img_1)
# print(type(img))
# print(img.shape)
