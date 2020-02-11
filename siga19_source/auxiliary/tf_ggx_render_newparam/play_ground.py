import numpy as np
import math
import cv2
import tensorflow as tf
from tf_ggx_render_utils import visualize_init,visualize_new

RENDER_SCALAR =  5*1e3/math.pi

if __name__ == "__main__":
    a = np.random.rand(2,3)
    idxs = np.tile(np.array([1,2]).reshape([1,2]),[2,1])
    x_axis_index=np.tile(np.arange(len(a)), (idxs.shape[1],1)).transpose()
    
    c = a[x_axis_index,idxs]
    print(a)
    print(idxs)
    print(x_axis_index)
    print(c)
        
