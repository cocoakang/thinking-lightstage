import numpy as np
import sys
import cv2
import math
sys.path.append('../utils/')
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_new
sys.path.append('./auxiliary/tf_ggx_render_newparam')
from tf_ggx_render import tf_ggx_render


RENDER_SCALAR = 5*1e3/math.pi
BOX_EDGE = 50.0
test_num = 10
if __name__ == "__main__":

    rendering_parameters = {}
    rendering_parameters["parameter_len"] = 10
    rendering_parameters["batch_size"] = 8
    rendering_parameters["lumitexel_size"] = 24576
    rendering_parameters["is_grey_scale"] = True
    rendering_parameters["config_dir"] = "../tf_ggx_render/tf_ggx_render_configs_1x1/"


    input_params = tf.placeholder(tf.float32,shape=[rendering_parameters["batch_size"],rendering_parameters["parameter_len"]],name="input_params")
    input_positions = tf.placeholder(tf.float32,shape=[rendering_parameters["batch_size"],3],name="render_positions")

    renderer = tf_ggx_render(rendering_parameters)

    ground_truth_lumitexels = renderer.draw_rendering_net(input_params,input_positions,"ground_truth_renderer")

    with tf.Session() as sess:
        current_pos = np.array([
            [BOX_EDGE,BOX_EDGE,BOX_EDGE],
            [BOX_EDGE,BOX_EDGE,-BOX_EDGE],
            [BOX_EDGE,-BOX_EDGE,BOX_EDGE],
            [BOX_EDGE,-BOX_EDGE,-BOX_EDGE],
            [-BOX_EDGE,BOX_EDGE,BOX_EDGE],
            [-BOX_EDGE,-BOX_EDGE,BOX_EDGE],
            [-BOX_EDGE,BOX_EDGE,-BOX_EDGE],
            [-BOX_EDGE,-BOX_EDGE,-BOX_EDGE]
        ],np.float32)

        test_param = np.array([
            [0.0,]
        ])
