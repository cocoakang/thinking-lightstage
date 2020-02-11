import tensorflow as tf
from check_quality_tamer import Tame_Tamer
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_init,visualize_new
from synthesis_data_generator import Synthesis_Data_Generator

UTILS_CONFIG_PATH = "G:/current_work/utils/"

def check_quality(result,log_path,global_step):
    img_path = log_path+"imgs_{}/".format(global_step)
    make_dir(img_path)
    np.savetxt(img_path+"param_ground_truth.csv",result[2],delimiter=',')
    # np.savetxt(img_path+"param_guessed.csv",result[3],delimiter=',')
    np.savetxt(img_path+"position_ground_truth.csv",result[3],delimiter=',')
    np.savetxt(img_path+"n_dot_view.csv",result[4].reshape([-1,1]),delimiter=',')
    np.savetxt(img_path+"view_dir.csv",result[5].reshape([-1,3]),delimiter=',')
    np.savetxt(img_path+"n.csv",result[6].reshape([-1,3]),delimiter=',')
    np.savetxt(img_path+"n_local.csv",result[7].reshape([-1,3]),delimiter=',')
    # np.savetxt(img_path+"param_origin.csv",result[5],delimiter=',')
    # np.savetxt(img_path+"position_guessed.csv",result[5],delimiter=',')
    for idx,a_gt_lumi in enumerate(result[0]):
        gt_lumi_img = visualize_new(a_gt_lumi.reshape([-1]),scalerf=255.0)
        guessed_lumi_img = visualize_new(np.exp(result[1][idx].reshape([-1]))-1,scalerf=255.0)
        cv2.imwrite(img_path+"{}_gt.png".format(idx),gt_lumi_img)
        cv2.imwrite(img_path+"{}_guessed.png".format(idx),guessed_lumi_img)
        np.savetxt(img_path+"{}_mm.csv".format(idx),result[2][idx].reshape([-1,1]),delimiter=',')

def gaussian_random_matrix(n_components, n_features, random_state=None):
    components = np.random.normal(loc=0.0,
                                  scale=1.0 / np.sqrt(n_components),
                                  # scale = 0.25,
                                  size=(n_components, n_features))
    return components

def getW(lumi_len, K):
    random = gaussian_random_matrix(lumi_len, K)
    random_reshape = np.reshape(random, (lumi_len, K))
    return random_reshape

MAX_ITR = 5000000
VALIDATE_ITR = 5
CHECK_QUALITY_ITR=5000
SAVE_MODEL_ITR=10000
LOG_ROOT="logs/"
if __name__ == "__main__":
    make_dir(LOG_ROOT)
    visualize_init(UTILS_CONFIG_PATH)
    data_pan = sys.argv[1]
    pretrained_model = sys.argv[2]
    ########################################
    ######step1 parse config
    ########################################
    train_configs = {}
    train_configs["parameter_len"] = 7#normal2 tangent1 axay2 pd1 ps1
    train_configs["lumitexel_length"] = 24576
    train_configs["measurements_length"] = 16
    train_configs["learning_rate"] = 1e-4
    train_configs["tamer_name"] = "tamer"
    train_configs["logPath"] = LOG_ROOT+"logs_lumitexel_guesser_iso/"
    make_dir(train_configs["logPath"])
    train_configs["pretrained_projection_matrix_path"] = "G:/2019_jointly_capture_training/3_7/Julia_feature_extractor_865000/"

    train_configs["batch_size"] = 60
    train_configs["data_pan"] = data_pan

    make_dir(train_configs["logPath"])
    log_details_dir = train_configs["logPath"]+"details/"
    train_configs["log_details_dir"] = log_details_dir
    make_dir(log_details_dir)
    myMine = Synthesis_Data_Generator(train_configs)
    ########################################
    ######step2 draw training net
    ########################################
    myTamer = Tame_Tamer(train_configs)
    print("[TRAIN]drawing net...")
    myTamer.draw_train_net("guess_net",projectionMatrix_trainable=True)
    myTamer.init()
    W = getW(train_configs["lumitexel_length"], train_configs["measurements_length"])
    # print(W.dtype)
    # W.tofile("logs/W.bin")
    # trained_projection_matrix = np.fromfile(train_configs["pretrained_projection_matrix_path"]+"W.bin",np.float32).reshape([train_configs["lumitexel_length"],train_configs["measurements_length"]])
    # W_pos = np.maximum(trained_projection_matrix,0)
    # W_neg = np.minimum(trained_projection_matrix,0)
    # trained_projection_matrix = np.concatenate([W_pos,W_neg*-1],axis = -1)

    myTamer.load_projection_matrix(W)
    myTamer.restore_model_assign(pretrained_model)
    print("[TRAIN]DONE.")

    ########################################
    ######step3 train net
    ########################################
    log_root = LOG_ROOT+"tmp/"
    log_measurments_dir = log_root+"measurements/"
    img_root = log_root+"imgs/"
    make_dir(log_root)
    make_dir(log_measurments_dir)
    make_dir(img_root)

    pf = open(log_root+"ae_output_data.bin","wb")
    counter = 0
    while True:
        tmpData,ifValid,num = myMine.gen_batch()
        if not ifValid:
            break
        result = myTamer.check_quality(tmpData)
        result[1][:num].astype(np.float32).tofile(pf)
        tmp_m = result[2].reshape([-1,3,train_configs["measurements_length"]])
        tmp_lumis = result[1][:num].reshape([-1,3,train_configs["lumitexel_length"]])
        num = tmp_m.shape[0]
        for idx in range(num):
            np.savetxt(log_measurments_dir+"{}_mm.csv".format(counter),tmp_m[idx].T,delimiter=',')
            img = []
            for lumi in tmp_lumis[idx]:
                img.append(visualize_new(lumi))
            img = np.array(img)
            img = np.transpose(img,axes=[1,2,0])
            img = img[:,:,::-1]
            cv2.imwrite(img_root+"{}.png".format(counter),img*255.0*0.5)
            counter+=1
        # break
    pf.close()
    print("[TAME]training done.")