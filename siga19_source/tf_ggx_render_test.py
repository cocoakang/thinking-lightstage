import numpy as np
import cv2
import math
import sys
sys.path.append("../utils/")
from lumitexel_related import visualize_init,visualize_new

lumitexel_size=24576
scaler = 5*1e3/math.pi

UTILS_CONFIG_PATH = "G:/current_work/utils/"

if __name__ == "__main__":
    visualize_init(UTILS_CONFIG_PATH)
    thread_id = 1
    parameters={}
    parameters["config_dir"] = "../tf_ggx_render/tf_ggx_render_configs_1x1/"
    
    root = "G:/no_where/test_rendering/"

    data = np.fromfile(root+"cpu_lumi.bin",np.float32).reshape([-1,2,lumitexel_size])
    data2 = np.fromfile(root+"tf_lumi.bin",np.float32).reshape([-1,2,lumitexel_size])
    for idx,a_lumi in enumerate(data):
        cv2.imshow("img_direct",visualize_new(a_lumi[0],scalerf=scaler))
        cv2.imshow("img_omega",visualize_new(a_lumi[1],scalerf=scaler))
        cv2.imshow("img2_direct",visualize_new(data2[idx][0],scalerf=1))
        cv2.imshow("img2_omega",visualize_new(data2[idx][1],scalerf=1))
        cv2.waitKey(0)

