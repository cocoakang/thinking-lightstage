import tensorflow as tf
import numpy as np
import sys
import cv2
import math
from tame_tamer import Tame_Tamer
sys.path.append("../utils/")
sys.path.append("../")
from tf_ggx_render.tf_ggx_render import tf_ggx_render
from lumitexel_related import visualize_init,visualize_new
from dir_folder_and_files import make_dir
UTILS_CONFIG_PATH = "G:/current_work/utils/"
if __name__ == "__main__":
    visualize_init(UTILS_CONFIG_PATH)
    data_root = "G:/2019_fresh_meat/3_29_beer_can/pattern/"
    log_root = data_root+"guess_log/"
    need_dump = True
    make_dir(log_root)
    GUESS_SCALAR = float("186.45603942871094")
    test_configs = {}
    test_configs["parameter_len"] = 7
    test_configs["lumitexel_length"] = 24576
    test_configs["measurements_length"] = 16
    test_configs["learning_rate"] = 1e-4
    test_configs["tamer_name"] = "tamer"
    test_configs["logPath"] = "logs/"
    test_configs["pretrained_projection_matrix_path"] = "G:/2019_jointly_capture_training/3_7/Julia_feature_extractor_865000/"

    test_configs["batch_size"] = 50

    test_configs["model_path"] = "G:/current_work/BRDF_param_guesser/logs/models/tamer_495000/tamer"


    input_measurements = tf.placeholder(tf.float32,shape=[test_configs["batch_size"],test_configs["measurements_length"]],name="input_measurements")
    input_positions = tf.placeholder(tf.float32,shape=[test_configs["batch_size"],3],name="input_positions")

    the_trained_tamer = Tame_Tamer(test_configs,for_train=False)
    with tf.variable_scope("guess_net",reuse=tf.AUTO_REUSE):
        guessed_params_node = the_trained_tamer.draw_param_guesser(input_measurements,"param_guesser")

    if need_dump:
        render_configs={}
        render_configs["parameter_len"] = 7
        render_configs["batch_size"] = test_configs["batch_size"]
        render_configs["lumitexel_size"] = 24576
        render_configs["is_grey_scale"] = True
        render_configs["config_dir"] = "../tf_ggx_render/tf_ggx_render_configs_1x1/"
        theRender = tf_ggx_render(render_configs)
        rendered_lumi_node = theRender.draw_rendering_net(guessed_params_node,input_positions,"test_render")
        
    with tf.Session() as sess:
        allVariables = tf.global_variables()
        reader = tf.train.NewCheckpointReader(test_configs["model_path"])
        if need_dump:
            init = tf.global_variables_initializer()
            sess.run(init)
        for a_var in allVariables:
            itsname = a_var.name[:-2]
            if "guess_net" in itsname:
                data = reader.get_tensor(itsname)
                a_var.load(data,sess)
                print("assign "+itsname+"...")    

        ##############################################test here
        measurements = np.fromfile(data_root+"cam00_data_32_nocc_compacted.bin",np.float32).reshape([-1,3,test_configs["measurements_length"]])*GUESS_SCALAR
        measurements_grey = np.mean(measurements,axis=1)#shape=[-1,16]
        
        data_ptr = 0
        f_test = open(log_root+"guessed_params.csv","w")
        with open(data_root+"guessed_params.bin","wb") as presf:
            while True:
                tmp_measurements = measurements_grey[data_ptr:data_ptr+test_configs["batch_size"]]
                valids = tmp_measurements.shape[0]
                data_ptr+=valids
                if tmp_measurements.shape[0] == 0:
                    break
                if tmp_measurements.shape[0] < test_configs["batch_size"]:
                    tmp_measurements = np.concatenate([tmp_measurements,np.zeros([test_configs["batch_size"]-tmp_measurements.shape[0],test_configs["measurements_length"]],np.float32)],axis=0)
                
                if need_dump:
                    tmp_guessed_params,tmp_rendered = sess.run([guessed_params_node,rendered_lumi_node],feed_dict={
                        input_measurements:tmp_measurements,
                        input_positions:np.zeros([test_configs["batch_size"],3],np.float32)
                    })
                    tmp_guessed_params = tmp_guessed_params[:valids]
                    tmp_guessed_params.tofile(presf)
                    np.savetxt(f_test,tmp_guessed_params,delimiter=',')
                    tmp_rendered = tmp_rendered[:valids]*5*1e3/math.pi
                    for idx_tmp,a_lumi in enumerate(tmp_rendered):
                        img = visualize_new(a_lumi,scalerf=255.0)
                        cv2.imwrite(log_root+"img_{}.png".format(data_ptr+idx_tmp),img)

                else:
                    tmp_guessed_params = sess.run(guessed_params_node,feed_dict={
                        input_measurements:tmp_measurements,
                        input_positions:np.zeros([test_configs["batch_size"],3],np.float32)
                    })
                    np.savetxt(f_test,tmp_guessed_params,delimiter=',')
                    tmp_guessed_params = tmp_guessed_params[:valids]
                    tmp_guessed_params.tofile(presf)
        f_test.close()
                


