import numpy as np
import cv2
import math

lumitexel_size=24576
scaler = 5*1e3/math.pi

def visualize_init(path="visualize_idxs_full.bin",lumitexel_size=24576):
    pf = open(path,"rb")
    print("initing idx....")
    global visualize_idxs
    visualize_idxs = np.fromfile(pf,dtype = np.int32).reshape([-1,2])
    if visualize_idxs.shape[0] != lumitexel_size :
        print("[VISUALIZE]:error dimension")
        exit()
    pf.close()
    print("done.")

def visualize_new(data,len = 64,scalerf=1.0):
    img = np.zeros([len * 3, len * 4],np.float32)
    for i in range(data.shape[0]):
        img[visualize_idxs[i][1]][visualize_idxs[i][0]] = data[i] * scalerf
    return img

if __name__ == "__main__":
    thread_id = 1
    parameters={}
    parameters["config_dir"] = './tf_ggx_render_configs/'
    visualize_init(parameters["config_dir"]+"visualize_idxs_full.bin")

    data = np.fromfile(parameters["config_dir"]+"test_rendered.bin",np.float32).reshape([-1,lumitexel_size])
    data2 = np.fromfile(parameters["config_dir"]+"test_rendered{}.bin".format(thread_id),np.float32).reshape([-1,lumitexel_size])
    for idx,a_lumi in enumerate(data):
        cv2.imshow("img",visualize_new(a_lumi,scalerf=scaler))
        cv2.imshow("img2",visualize_new(data2[idx],scalerf=scaler))
        cv2.waitKey(0)

