import numpy as np
import cv2
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_init,visualize_new

UTILS_CONFIG_PATH = "G:/current_work/utils/"

if __name__ == "__main__":
    data_root=sys.argv[1]
    data_name=sys.argv[2]

    visualize_init(UTILS_CONFIG_PATH)

    data = np.fromfile(data_root+data_name,np.float32).reshape([-1,3,24576])

    img_root = data_root+"img/"
    make_dir(img_root)
    for idx,apixel in enumerate(data):
        r = np.expand_dims(visualize_new(apixel[0]),axis=2)
        g = np.expand_dims(visualize_new(apixel[1]),axis=2)
        b = np.expand_dims(visualize_new(apixel[2]),axis=2)

        img = np.concatenate([b,g,r],axis=-1)
        cv2.imwrite(img_root+"{}.png".format(idx),img*255)