import tensorflow as tf
from tame_tamer import Tame_Tamer
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_init,visualize_new,visualize_cube_slice_init,visualize_cube_slice
import math
from net_parameter_generator import Net_Parameter_Generator

UTILS_CONFIG_PATH = "../utils/"

def check_quality(result,log_path,global_step):
    pass
    # img_path = log_path+"imgs_{}/".format(global_step)
    # make_dir(img_path)
    # np.savetxt(img_path+"param_ground_truth.csv",result[2],delimiter=',')
    # # np.savetxt(img_path+"param_guessed.csv",result[3],delimiter=',')
    # np.savetxt(img_path+"position_ground_truth.csv",result[3],delimiter=',')
    
    # np.savetxt(img_path+"n_gt.csv",result[6].reshape([-1,3]),delimiter=',')
    # np.savetxt(img_path+"n_nn.csv",result[7].reshape([-1,3]),delimiter=',')
    # np.savetxt(img_path+"pd_gt.csv",result[8].reshape([-1,1]),delimiter=',')
    # np.savetxt(img_path+"pd_nn.csv",result[9].reshape([-1,1]),delimiter=',')
    # np.savetxt(img_path+"param_origin.csv",result[5],delimiter=',')
    # np.savetxt(img_path+"position_guessed.csv",result[5],delimiter=',')
    # for idx,a_gt_lumi in enumerate(result[0]):
    #     gt_lumi_img = visualize_new(a_gt_lumi.reshape([-1]),scalerf=255.0)
    #     gt_lumi_rotated_img = visualize_new(result[4][idx].reshape([-1]),scalerf=255.0)
    #     guessed_lumi_img = visualize_new(np.exp(result[1][idx].reshape([-1]))-1,scalerf=255.0)
    #     cv2.putText(gt_lumi_img, '{:.2f}'.format(result[5][idx][0]/math.pi*180.0), (128,64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
    #     cv2.imwrite(img_path+"{}_gt.png".format(idx),gt_lumi_img)
    #     # print(result[5][idx]/math.pi*360.0)
    #     cv2.imwrite(img_path+"{}_gt_rotated.png".format(idx),gt_lumi_rotated_img)
    #     cv2.imwrite(img_path+"{}_guessed.png".format(idx),guessed_lumi_img)

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
DUMP_ITR = 250000
VALIDATE_ITR = 5
CHECK_QUALITY_ITR=5000
SAVE_MODEL_ITR=10000
LOG_ROOT="logs/"
if __name__ == "__main__":
    make_dir(LOG_ROOT)
    visualize_init(UTILS_CONFIG_PATH)
    visualize_cube_slice_init(UTILS_CONFIG_PATH,64)
    visualize_cube_slice_init(UTILS_CONFIG_PATH,8)
    data_root = sys.argv[1]
    ########################################
    ######step1 parse config
    ########################################
    train_configs = {}
    train_configs["DUMP_ITR"] = DUMP_ITR
    train_configs["parameter_len"] = 7#normal2 tangent1 axay2 pd1 ps1
    train_configs["lumitexel_length"] = 24576
    train_configs["loss_with_form_fractor"] = False
    train_configs["measurements_length"] = 6
    train_configs["learning_rate"] = 1e-4
    train_configs["rotate_num"] = 12
    train_configs["pre_load_buffer_size"] = 500000
    train_configs["tamer_name"] = "tamer"
    train_configs["logPath"] = LOG_ROOT+"siga19/"
    make_dir(train_configs["logPath"])
    
    train_configs["batch_size"] = 50
    train_configs["data_root"] = data_root
    train_configs["with_length_predict"] = False

    lambdas = {}
    lambdas["pd"] =1.0
    lambdas["ps"] = 1e-2
    lambdas["ps_length"] = 1.0
    lambdas["normal"] = 1.0
    lambdas["p"]=1e-3
    train_configs["lambdas"] = lambdas

    loss_configs = {}
    loss_configs["use_log"] = False
    loss_configs["use_weight"] = False
    loss_configs["use_dot"] = False
    loss_configs["use_l2"] = True
    train_configs["loss_configs"] = loss_configs


    make_dir(train_configs["logPath"])
    log_details_dir = train_configs["logPath"]+"details/"
    train_configs["log_details_dir"] = log_details_dir
    make_dir(log_details_dir)
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
    # W = trained_projection_matrix
    # W_pos = np.maximum(trained_projection_matrix,0)
    # W_neg = np.minimum(trained_projection_matrix,0)
    # trained_projection_matrix = np.concatenate([W_pos,W_neg*-1],axis = -1)

    myTamer.load_projection_matrix(W)
    # restore_step = 80000
    # myTamer.restore_model(train_configs["logPath"]+"models/tamer_{}/tamer".format(restore_step))
    # myTamer.load_use_assign(train_configs["logPath"]+"models/tamer_{}/tamer".format(restore_step))
    # myTamer.rest
    print("[TRAIN]DONE.")
    trainMine = Net_Parameter_Generator(train_configs,"train")
    valMine = Net_Parameter_Generator(train_configs,"val")

    ########################################
    ######step3 train net
    ########################################
    for global_step in range(MAX_ITR):
        if global_step % 1000 == 0:
            print("[TAME] global step:",global_step)
        if global_step % VALIDATE_ITR == 0:
            val_data = valMine.generate_validating_data()
            myTamer.validate(val_data)
        if global_step % CHECK_QUALITY_ITR == 0:
            tmp_log_path = log_details_dir+"{}/".format(global_step)
            make_dir(tmp_log_path)
            # myTamer.saveWLog(global_step,tmp_log_path)
            val_data = valMine.generate_validating_data()
            result = myTamer.check_quality(val_data)
            check_quality(result,tmp_log_path,global_step)
        if global_step % SAVE_MODEL_ITR == 0 and global_step != 0:
            myTamer.save_model()

        training_data = trainMine.generate_training_data()
        myTamer.train(training_data)
        # break
    print("[TAME]training done.")
