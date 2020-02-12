import tensorflow as tf
import sys
sys.path.append('../utils/')
import math
import shutil
import numpy as np
import cv2
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_new,visualize_cube_slice,expand_img
sys.path.append('./auxiliary/tf_ggx_render_newparam')
from tf_ggx_render import tf_ggx_render

class Tame_Tamer:
    def __init__(self,train_configs,for_train=True):
        self.end_points = {}
        if for_train:
            self.DUMP_ITR = train_configs["DUMP_ITR"]
            self.parameter_len = train_configs["parameter_len"]
            self.lumitexel_length = train_configs["lumitexel_length"]
            self.slice_sample_num_pd = 8
            self.slice_length_pd = self.slice_sample_num_pd*self.slice_sample_num_pd*6
            self.slice_sample_num_ps = 64
            self.slice_length_ps = self.slice_sample_num_ps*self.slice_sample_num_ps*6

            self.lambdas = train_configs["lambdas"]

            self.loss_configs = train_configs["loss_configs"]

            self.measurements_length = train_configs["measurements_length"]
            self.learning_rate = train_configs["learning_rate"]
            self.tamer_name = train_configs["tamer_name"]
            self.loss_with_form_fractor = train_configs["loss_with_form_fractor"]
            self.logPath = train_configs["log_details_dir"]
            make_dir(self.logPath)
            self.modelPath = train_configs["logPath"]+"models/"
            make_dir(self.modelPath)
            self.batch_size = train_configs["batch_size"]#*train_configs["rotate_num"]

            standard_rendering_parameters = {}

            standard_rendering_parameters["parameter_len"] = self.parameter_len
            standard_rendering_parameters["batch_size"] = self.batch_size
            standard_rendering_parameters["lumitexel_size"] = self.lumitexel_length
            standard_rendering_parameters["is_grey_scale"] = True
            standard_rendering_parameters["config_dir"] = "auxiliary/tf_ggx_render_newparam/tf_ggx_render_configs_1x1/"
            self.end_points["standard_rendering_parameters"] = standard_rendering_parameters
            self.with_length_predict = train_configs["with_length_predict"]

            self.RENDER_SCALAR = 5*1e3/math.pi
            self.SLICE_SCALAR = 5*1e3/math.pi
        else:
            self.parameter_len = train_configs["parameter_len"]

    def draw_normal_net(self,measurements,scope_name,renderer,isTraining,part_trainable = True,af = tf.nn.leaky_relu):
        '''
        measurements = [batch,measure_ments]
        return =[batch,param_length]

        scope_name: a string, the name of this part
        '''
        self.ae_trainable = part_trainable
        endPoints = {}
        size = 256
        keep_prob = 0.90
        namescope = tf.variable_scope(scope_name)
        withBN = False
        # intputData = [measurements]
        x_n = measurements
        x_n = tf.nn.l2_normalize(x_n, dim=-1)
        with namescope:
            print("fcNet!")
            ###for input###
            ###start####
            block_ns = tf.variable_scope('fc_kblock1')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 512
                x_n = tf.contrib.layers.fully_connected(x_n, size, trainable=self.ae_trainable,
                activation_fn=af,scope = tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)

            block_ns = tf.variable_scope('fc_block2')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 512
                x_n = tf.contrib.layers.fully_connected(x_n, size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)

            block_ns = tf.variable_scope('fc_block3')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 256
                x_n = tf.contrib.layers.fully_connected(x_n, size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)
            
            block_ns = tf.variable_scope('fc_block4')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 256
                x_n = tf.contrib.layers.fully_connected(x_n, size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)

            block_ns = tf.variable_scope('fc_block5')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 128
                x_n = tf.contrib.layers.fully_connected(x_n, size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)

            ##medium start###
            block_ns = tf.variable_scope('fc_block6')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                size = 64
                x_n = tf.contrib.layers.fully_connected(x_n, size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                x_n = tf.contrib.layers.dropout(x_n, keep_prob=keep_prob, is_training=isTraining)

            block_ns = tf.variable_scope('fc_block7')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                x_n = tf.contrib.layers.fully_connected(x_n, 3, activation_fn=None,trainable=self.ae_trainable, scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
            
            x_n = tf.nn.l2_normalize(x_n, dim=-1)
            # print("lumi shape = ", lumi.shape)
        return x_n
    
    def draw_param_guesser(self,measurements,scope_name,renderer,isTraining,part_trainable = True,af = tf.nn.relu):
        '''
        measurements = [batch,measure_ments]
        return =[batch,param_length]

        scope_name: a string, the name of this part
        '''
        self.ae_trainable = True
        af = tf.nn.leaky_relu
        endPoints = {}
        size = 256
        keep_ratio = 0.90
        namescope = tf.variable_scope(scope_name)
        withBN = False
        intputData = [measurements]
        with namescope:
            print("fcNet!")
            ###for input###
            ###start####
            block_ns = tf.variable_scope('fc_kblock1')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                x_n = []
                print("draw fc block 1")
                for tmp in intputData:
                    x_n_tmp = tf.contrib.layers.fully_connected(tmp, size, trainable=self.ae_trainable,
                    activation_fn=af,scope = tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n_tmp = tf.contrib.layers.dropout(x_n_tmp,keep_prob=keep_ratio,is_training = isTraining)
                    x_n.append(x_n_tmp)
            endPoints["firstFC_end"] = x_n[-1]
            
            block_ns = tf.variable_scope('fc_block1_2')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 1_2")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 256
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block1_3')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 1_3")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 256
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            back_for_ps = x_n.copy()
            block_ns = tf.variable_scope('fc_block2')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 2")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 256
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            position_bak = x_n.copy()
            block_ns = tf.variable_scope('fc_block3')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 3")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn2",x_n,size,isTraining,af,self.ae_trainable)
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block4')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 4")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn3",x_n,size,isTraining,af,self.ae_trainable)
                size = 256
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            medium_x_n = x_n.copy()
            block_ns = tf.variable_scope('fc_block8_pd')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 8")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size=128
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)               
            ##deep end###
            block_ns = tf.variable_scope('deep_output_pd')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw output decoder")
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], self.slice_length_pd, activation_fn=None, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
            guessed_pd = x_n[0]

            #########################################
            #########ps lobe
            #########################################
            block_ns = tf.variable_scope('fc_block3_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 3")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn4",back_for_ps,size,isTraining,af,self.ae_trainable)
                size = 512
                for idx,value in enumerate(back_for_ps):
                    x_n[idx] = tf.contrib.layers.fully_connected(back_for_ps[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block4_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 4")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 1024
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block5_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 5")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn4",medium_x_n,size,isTraining,af,self.ae_trainable)
                size = 2048
                for idx,value in enumerate(medium_x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block6_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 6")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 4096
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            back_x_n =x_n[0]
            ##deep start###
            block_ns = tf.variable_scope('fc_block7_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 7")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block8_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 8")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)               
            ##deep end###
            block_ns = tf.variable_scope('deep_output_ps')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw output decoder")
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], self.slice_length_ps, activation_fn=None, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
            guessed_ps = x_n[0]

            if self.with_length_predict:
                with tf.variable_scope('ps_value1'):
                    print("draw ps value")
                    ps_predicted = tf.contrib.layers.fully_connected(back_x_n,2048,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
                with tf.variable_scope('ps_value2'):
                    print("draw ps value")
                    ps_predicted = tf.contrib.layers.fully_connected(ps_predicted,1024,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
                with tf.variable_scope('ps_value3'):
                    print("draw ps value")
                    ps_predicted = tf.contrib.layers.fully_connected(ps_predicted,512,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
                with tf.variable_scope('ps_value4'):
                    print("draw ps value")
                    ps_predicted = tf.contrib.layers.fully_connected(ps_predicted,256,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
                with tf.variable_scope('ps_value5'):
                    ps_predicted = tf.contrib.layers.fully_connected(ps_predicted,1,activation_fn=None,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
            else:
                ps_predicted = tf.zeros([self.batch_size,1],tf.float32)

            with tf.variable_scope('p_value1'):
                print("draw p value")
                p_predicted = tf.contrib.layers.fully_connected(position_bak,512,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
            with tf.variable_scope('p_value2'):
                print("draw p value")
                p_predicted = tf.contrib.layers.fully_connected(p_predicted,512,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
            with tf.variable_scope('p_value3'):
                print("draw p value")
                p_predicted = tf.contrib.layers.fully_connected(p_predicted,512,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
            with tf.variable_scope('p_value4'):
                print("draw p value")
                p_predicted = tf.contrib.layers.fully_connected(p_predicted,256,activation_fn=af,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)
            with tf.variable_scope('p_value5'):
                p_predicted = tf.contrib.layers.fully_connected(p_predicted,3,activation_fn=None,trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse=tf.AUTO_REUSE)


        return guessed_pd,guessed_ps,ps_predicted,p_predicted

    def draw_train_net(self,net_name,projectionMatrix_trainable):
        '''
        net_name: a string,the name of this net
        '''
        with tf.variable_scope(net_name,reuse=tf.AUTO_REUSE):
            #[1]define input parameters
            input_params = tf.placeholder(tf.float32,shape=[self.batch_size,self.parameter_len],name="input_params")
            self.end_points["input_params"] = input_params
            input_positions = tf.placeholder(tf.float32,shape=[self.batch_size,3],name="render_positions")
            self.end_points["input_positions"] = input_positions
            input_rotate_angles = tf.placeholder(tf.float32,shape=[self.batch_size,1],name="rotate_angles")
            self.end_points["input_rotate_angles"] = input_rotate_angles
            isTraining = tf.placeholder(tf.bool,name = "isTraining_ph")
            self.end_points["istraining"] = isTraining
            
            #[2]draw rendering net,render ground truth 
            renderer = tf_ggx_render(self.end_points["standard_rendering_parameters"])
            self.renderer = renderer

            slice_rendering_parameters = self.end_points["standard_rendering_parameters"].copy()
            slice_rendering_parameters["config_dir"] = "auxiliary/tf_ggx_render_newparam/tf_ggx_render_configs_cube_slice_64x64/"
            slice_rendering_parameters["lumitexel_size"] = self.slice_length_ps
            renderer_slice_ps = tf_ggx_render(slice_rendering_parameters)

            slice_rendering_parameters = self.end_points["standard_rendering_parameters"].copy()
            slice_rendering_parameters["config_dir"] = "auxiliary/tf_ggx_render_newparam/tf_ggx_render_configs_cube_slice_8x8/"
            slice_rendering_parameters["lumitexel_size"] = self.slice_length_pd
            renderer_slice_pd = tf_ggx_render(slice_rendering_parameters)
            
            rotate_theta_zero = tf.zeros([self.batch_size,1],tf.float32)
            ground_truth_lumitexels_direct = renderer.draw_rendering_net(input_params,input_positions,rotate_theta_zero,"ground_truth_renderer_direct")#[batch,lightnum,1]
            ground_truth_lumitexels_direct *= self.RENDER_SCALAR
            self.end_points["ground_truth_lumitexels_direct"] = ground_truth_lumitexels_direct

            ground_truth_slice_ps = renderer_slice_ps.draw_rendering_net(input_params,input_positions,rotate_theta_zero,"ground_truth_renderer_cube_slice_ps",pd_ps_wanted="ps_only")
            ground_truth_slice_ps = ground_truth_slice_ps*self.SLICE_SCALAR
            self.end_points["ground_truth_slice_ps"] = ground_truth_slice_ps    

            ground_truth_slice_pd = renderer_slice_pd.draw_rendering_net(input_params,input_positions,rotate_theta_zero,"ground_truth_renderer_cube_slice_pd",pd_ps_wanted="pd_only")
            ground_truth_slice_pd = ground_truth_slice_pd*self.SLICE_SCALAR
            self.end_points["ground_truth_slice_pd"] = ground_truth_slice_pd      

            
            # n_local,_ = renderer.get_n("ground_truth_renderer_direct")
            # self.end_points["n_local_gt"] = n_local
            # pd = renderer.get_pd("ground_truth_renderer_direct")
            # self.end_points["pd_gt"] = pd

            #[3]draw linear projection layer
            with tf.variable_scope('linear_projection',reuse = tf.AUTO_REUSE):
                kernel = tf.get_variable(name = 'projection_matrix',
                                        trainable=projectionMatrix_trainable,
                                        shape = [self.lumitexel_length,self.measurements_length],
                                        initializer=tf.contrib.layers.xavier_initializer()
                                        )#[lightnum,measure_ments]
                self.end_points['projection_matrix_origin'] = kernel
                kernel = tf.transpose(kernel,perm=[1,0])
                kernel = tf.nn.l2_normalize(kernel,axis=1)
                kernel = tf.transpose(kernel,perm=[1,0])
                self.end_points['projection_matrix'] = kernel


                measure_ments = tf.matmul(tf.squeeze(ground_truth_lumitexels_direct,axis=2),kernel)#[batch,measure_ments]
            
            #[3.1]noiser
            with tf.variable_scope("noiser"):
                noise = tf.random_normal(tf.shape(measure_ments),mean = 0.0,stddev = 0.01,name="noise_generator")+1.
                measure_ments_noised = tf.multiply(measure_ments,noise)
            
            # measure_ments_noised_theta = tf.concat([measure_ments_noised,tf.sin(input_rotate_angles),tf.cos(input_rotate_angles)],axis=-1)
            #[4]draw guesser net
            guessed_lumitexels_pd,guessed_lumitexels_ps,ps_predicted,p_predicted = self.draw_param_guesser(measure_ments_noised,"param_guesser",renderer,isTraining)#[batch,param_length]
            
            self.end_points["ps_predicted"] = ps_predicted
            
            guessed_lumitexels_pd = tf.expand_dims(guessed_lumitexels_pd,axis=-1)
            self.end_points["guessed_lumitexels_pd"] = guessed_lumitexels_pd

            guessed_lumitexels_ps = tf.expand_dims(guessed_lumitexels_ps,axis=-1)
            self.end_points["guessed_lumitexels_ps"] = tf.exp(guessed_lumitexels_ps)-1.0


            predicted_normal = self.draw_normal_net(measure_ments_noised,"normal_net",renderer,isTraining)
            _,label_normal = renderer.get_n("ground_truth_renderer_direct")

            self.end_points["wi"] = renderer.get_wi("ground_truth_renderer_direct")
            self.end_points["light_dir"] = self.end_points["wi"]
            self.end_points["pos"] = input_positions
            self.end_points["view_dir"] = renderer.get_compute_node("ground_truth_renderer_direct","view_dir_rotated")
            self.end_points["normal"] = renderer.get_compute_node("ground_truth_renderer_direct","normal")


            # self.end_points["guessed_normal"] = normal_guessed
            # self.end_points["guessed_pd"] = pd_guessed

            # guessed_n,guessed_t,guessed_axay,guessed_pd,guessed_ps = tf.split(guessed_params,[3,3,2,1,1],axis=-1)
            # guessed_axay = tf.clip_by_value(guessed_axay,0.006,0.503)
            # guessed_pd = tf.clip_by_value(guessed_pd,0.0,1.0)
            # guessed_ps = tf.clip_by_value(guessed_ps,0.0,10.0)
            # guessed_params = tf.concat([guessed_n,guessed_t,guessed_axay,guessed_pd,guessed_ps],axis=-1)
            # self.end_points["guessed_params"] = guessed_params
            # # self.end_points["guessed_position"] = guessed_position
            # #[5]render guessed lumitexel
            # guessed_lumitexels = renderer.draw_rendering_net(guessed_params,input_positions,"guessed_renderer")#[batch,lightnum,1]
            # # guessed_lumitexels = renderer.draw_rendering_net(guessed_params,input_positions,"guessed_renderer")#[batch,lightnum,1]
            # guessed_lumitexels = guessed_lumitexels*self.RENDER_SCALAR
            # self.end_points["guessed_lumitexels"] = guessed_lumitexels
            
        # _,self.end_points["n_dot_view"],self.end_points["view_dir"],self.end_points["n"],self.end_points["n_local"] = renderer.get_n_dot_view_penalty("ground_truth_renderer")
        #[6]define loss

        ##########################
        ######pd loss
        #########################
        l2_loss_pd = tf.nn.l2_loss(guessed_lumitexels_pd-ground_truth_slice_pd,"l2_loss_pd")
        self.end_points["l2_loss_pd"] = l2_loss_pd

        ##########################
        ######ps loss
        #########################
        #----------part 1
        if self.loss_configs["use_weight"]:
            print("nonono")
            exit(-1)
            assert self.loss_configs["use_log"] == False
            ground_truth_slice_ps_tmp = tf.reshape(ground_truth_slice_ps,[self.batch_size,self.slice_length_ps])
            weights = 1.0/tf.reduce_sum(tf.square(ground_truth_slice_ps_tmp),axis=-1,keepdims=True)#[batch,1]

            subed = ground_truth_slice_ps-guessed_lumitexels_ps
            subed = tf.reshape(subed,[self.batch_size,self.slice_length_ps])
            loss_tmp = tf.reduce_sum(tf.square(subed),axis=-1,keepdims=True)*weights
            l2_loss_ps = tf.reduce_sum(loss_tmp)/2.0

        elif self.loss_configs["use_dot"]:
            print("nonono")
            exit(-1)
            ground_truth_slice_ps_tmp = tf.reshape(ground_truth_slice_ps,[self.batch_size,self.slice_length_ps])
            guessed_slice_ps_tmp = tf.reshape(guessed_lumitexels_ps,[self.batch_size,self.slice_length_ps])
            
            ground_truth_slice_ps_tmp = tf.nn.l2_normalize(ground_truth_slice_ps_tmp,axis=-1)
            guessed_slice_ps_tmp = tf.nn.l2_normalize(guessed_slice_ps_tmp,axis=-1)
            l2_loss_ps = tf.reduce_sum(1.0-tf.reduce_sum(tf.multiply(ground_truth_slice_ps_tmp,guessed_slice_ps_tmp),axis=-1))

        elif self.loss_configs["use_log"]:
            print("nonono")
            exit(-1)
            l2_loss_ps = tf.nn.l2_loss(guessed_lumitexels_ps-tf.maximum(tf.log(1.0+ground_truth_slice_ps),0.0),"l2_loss_ps")
        elif self.loss_configs["use_l2"]:
            # ground_truth_slice_ps_tmp = tf.reshape(ground_truth_slice_ps,[self.batch_size,self.slice_length_ps])
            # guessed_slice_ps_tmp = tf.reshape(guessed_lumitexels_ps,[self.batch_size,self.slice_length_ps])
            
            # ground_truth_slice_ps_tmp = tf.nn.l2_normalize(ground_truth_slice_ps_tmp,axis=-1)
            # alpha = 1e3
            ground_truth_slice_ps_tmp = tf.log(1.0+ground_truth_slice_ps)
            guessed_slice_ps_tmp = guessed_lumitexels_ps
            self.end_points["guessed_lumitexels_ps_normalized"] = guessed_slice_ps_tmp
            self.end_points["ground_truth_slice_ps_normalized"] = ground_truth_slice_ps_tmp
            l2_loss_ps = tf.nn.l2_loss(ground_truth_slice_ps_tmp-guessed_slice_ps_tmp,"l2_loss_ps")

        else:
            print("[TAMER]undefined l2 loss of ps")
        self.end_points["l2_loss_ps"] = l2_loss_ps

        #------------part 2
        tmp = tf.squeeze(ground_truth_slice_ps_tmp,axis=2)
        ground_truth_slice_length = tf.sqrt(tf.reduce_sum(tf.square(tmp),axis=-1,keepdims=True))
        ground_truth_slice_length = tf.log(1+ground_truth_slice_length)
        self.end_points["ground_truth_slice_length"] = ground_truth_slice_length
        if self.with_length_predict:
            l2_loss_length = tf.nn.l2_loss(ps_predicted-ground_truth_slice_length,"l2_loss_normalized")
        else:
            l2_loss_length = 0.0
        self.end_points["l2_loss_length"] = l2_loss_length

        #-------------part 3
        l2_loss_p = tf.nn.l2_loss(p_predicted-input_positions,"l2_loss_p")
        self.end_points["p_loss"] = l2_loss_p

        ##########################
        ######normal loss
        #########################
        l2_loss_normal = tf.nn.l2_loss(predicted_normal - label_normal)
        self.end_points["l2_loss_normal"] = l2_loss_normal

        l2_loss = self.lambdas["pd"]*l2_loss_pd+self.lambdas["ps"]*l2_loss_ps+self.lambdas["ps_length"]*l2_loss_length+self.lambdas["normal"]*l2_loss_normal+self.lambdas["p"]*l2_loss_p

        # l2_loss_normal = tf.nn.l2_loss(n_local-normal_guessed,"l2_loss_normal")
        # self.end_points["l2_loss_normal"] = l2_loss_normal

        # l2_loss_pd = tf.nn.l2_loss(pd-pd_guessed,"l2_loss_pd")
        # self.end_points["l2_loss_pd"]=l2_loss_pd

        # l2_loss_param = tf.nn.l2_loss((input_params-guessed_params)*np.array([1,1,1,0,0,0,0,0,1,0.1]),"l2_loss_param")*1e3
        # l2_loss_pos = tf.nn.l2_loss((input_positions-guessed_position)*np.array([0.02,0.02,0.02]),"l2_loss_position")*1e3
        # l2_loss = tf.nn.l2_loss(tf.maximum(tf.log(1.0+guessed_lumitexels),0.0)-tf.maximum(tf.log(1.0+ground_truth_lumitexels),0.0),"l2_loss")
        # print(l2_loss)
        # n_3d,tangent_3d,ax,ay,pd,ps = tf.split(guessed_params,[3,3,1,1,1,1],axis=1)
        loss_k = 1e5
        if projectionMatrix_trainable:
            regularization_loss = renderer.regularizer_relu(kernel,-1.0,1.0,loss_k)
                                # renderer.regularizer_relu(guessed_position,-50.0,50.0,loss_k)+\    
                                # n_dot_view_penalty+\
                                # renderer.regularizer_relu(theta,0.0,math.pi,loss_k)+\
        else:
            regularization_loss = 0

        total_loss = l2_loss+regularization_loss#+l2_loss_normal+l2_loss_pd#+l2_loss_param+l2_loss_pos

        # all_trainable_variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        # for a_variable in all_trainable_variables:
        #     print(a_variable)
        # exit()

        #[7]define optimizer
        global_step = tf.Variable(0, name='global_step',trainable=False)
        self.end_points["global_step"] = global_step
        learning_rate =tf.Variable(self.learning_rate, name='learning_rate',trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer
            optimizer = optimizer(learning_rate)
            self.trainHandler = optimizer.minimize(total_loss,global_step=global_step)
            self.end_points["optimizer"] = optimizer
        
        #[8]define others
        #[8.1] writer
        train_penalty_summary = tf.summary.scalar("loss_penalty_"+self.tamer_name,regularization_loss)
        train_l2_loss_summary = tf.summary.scalar("loss_l2_total_train_"+self.tamer_name,l2_loss)

        train_l2_loss_pd_summary = tf.summary.scalar("loss_pd_train_"+self.tamer_name,l2_loss_pd)
        train_l2_loss_pslength_summary = tf.summary.scalar("loss_pslength_train_"+self.tamer_name,l2_loss_length)
        train_l2_loss_ps_summary = tf.summary.scalar("loss_ps_train_"+self.tamer_name,l2_loss_ps)
        train_l2_loss_normal_summary = tf.summary.scalar("loss_normal_train_"+self.tamer_name,l2_loss_normal)
        train_l2_loss_p_summary = tf.summary.scalar("loss_p_train_"+self.tamer_name,l2_loss_p)
        
        val_l2_loss_summary = tf.summary.scalar("loss_l2_total_val_"+self.tamer_name,l2_loss)

        val_l2_loss_pd_summary = tf.summary.scalar("loss_pd_val_"+self.tamer_name,l2_loss_pd)
        val_l2_loss_pslength_summary = tf.summary.scalar("loss_pslength_val_"+self.tamer_name,l2_loss_length)
        val_l2_loss_ps_summary = tf.summary.scalar("loss_ps_val_"+self.tamer_name,l2_loss_ps)
        val_l2_loss_normal_summary = tf.summary.scalar("loss_normal_val_"+self.tamer_name,l2_loss_normal)
        val_l2_loss_p_summary = tf.summary.scalar("loss_p_val_"+self.tamer_name,l2_loss_p)
        # val_normal_l2_loss_summary = tf.summary.scalar("loss_normal_val_"+self.tamer_name,l2_loss_normal)
        # val_pd_l2_loss_summary = tf.summary.scalar("loss_pd_val_"+self.tamer_name,l2_loss_pd)
        
        

        tmp_zero_lumi = np.zeros(self.lumitexel_length,np.float32)
        tmp_zero_lumi_img = visualize_new(tmp_zero_lumi.reshape([-1]))
    
        zeros_lumis_init = tf.zeros_initializer()#tf.constant_initializer(zeros_lumis)

        self.check_images = tf.get_variable(name = 'check_pic',
                                        trainable=False,
                                        shape = [self.batch_size,tmp_zero_lumi_img.shape[0],tmp_zero_lumi_img.shape[1]*5,1],
                                        initializer=zeros_lumis_init,
                                        dtype=tf.uint8
                                        )#[lightnum,measure_ments]
        
        self.pattern_check = tf.get_variable(name = 'patterns',
                                        trainable=False,
                                        shape = [self.measurements_length,tmp_zero_lumi_img.shape[0],tmp_zero_lumi_img.shape[1],3],
                                        initializer=zeros_lumis_init,
                                        dtype=tf.uint8
                                        )

        # self.check_formfactor = tf.get_variable(name = 'check_pic_ff',
        #                                 trainable=False,
        #                                 shape = [self.batch_size,tmp_zero_lumi_img.shape[0],tmp_zero_lumi_img.shape[1],1],
        #                                 initializer=tf.zeros_initializer(),
        #                                 dtype=tf.uint8
        #                                 )#[lightnum,measure_ments]

        check_pic_summary = tf.summary.image("lumitexels",self.check_images,max_outputs=self.batch_size)
        pattern_summary = tf.summary.image("patterns",self.pattern_check,max_outputs=self.measurements_length)
        # check_pic_ff_summary = tf.summary.image("ff",self.check_formfactor,max_outputs=self.batch_size)
        
        
        #loss of matcher
        train_summary = tf.summary.merge([
            train_l2_loss_summary,train_penalty_summary,train_l2_loss_pslength_summary,train_l2_loss_pd_summary,train_l2_loss_ps_summary,train_l2_loss_normal_summary,train_l2_loss_p_summary
        ])
        val_summary = tf.summary.merge([
            val_l2_loss_summary,val_l2_loss_pd_summary,val_l2_loss_pslength_summary,val_l2_loss_ps_summary,val_l2_loss_normal_summary,val_l2_loss_p_summary
        ])
        check_summary = tf.summary.merge([
            check_pic_summary,pattern_summary#,check_pic_ff_summary
        ])
        self.end_points["train_summary"] = train_summary
        self.end_points["val_summary"] = val_summary
        self.end_points["check_summary"] = check_summary

        #[8.2] initializer
        init = tf.global_variables_initializer()
        self.end_points['init'] = init
        #[8.3] saver
        saver = tf.train.Saver()
        self.end_points['saver'] = saver
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config)
        
        writer = tf.summary.FileWriter(self.logPath, tf.get_default_graph())
        self.end_points['writer'] = writer

    def init(self):
        self.sess.run(self.end_points['init'])

    def train(self,training_data):
        _,summary,global_step = self.sess.run(
            [
                self.trainHandler,
                self.end_points['train_summary'],
                self.end_points["global_step"]
            ],
            feed_dict={
                self.end_points["input_params"]:training_data[0],
                self.end_points["input_positions"]:training_data[1],
                self.end_points["istraining"]:True,
                self.end_points["input_rotate_angles"]:training_data[2]
            }
        )
        
        self.end_points["writer"].add_summary(summary,global_step)

    def saveWLog(self,globalStep,tmp_log_path):
        W = self.sess.run(self.end_points['projection_matrix'])
        W_vector = W.reshape([-1])
        f = open(tmp_log_path+"W_{}.csv".format(globalStep),"w")
        [f.write("{},\n".format(W_vector[i]))  for i in range(W_vector.shape[0])]
        f.close()
        W = W.T
        for i in range(W.shape[0]):
            one_pattern = visualize_new(W[i]/W[i].max(),scalerf=255)#cook10240(W[i])#visualize(W[i]/W[i].max(),scalerf=255)
            one_pattern = np.expand_dims(one_pattern,axis = -1)
            one_pattern_pos = np.maximum(one_pattern,0.0)
            one_pattern_neg = np.minimum(one_pattern,0.0)*-1
            one_pattern = np.concatenate([one_pattern_neg,np.zeros(one_pattern.shape),one_pattern_pos],axis = -1)
            # one_pattern /= one_pattern.max()
            cv2.imwrite(tmp_log_path+"{}.png".format(i),one_pattern)

    def restore_model(self,path):
        print("loading model...")
        self.end_points['saver'].restore(self.sess,path)
        print("done.")

    def restore_global_step(self,global_step):
        self.end_points["global_step"].load(global_step,self.sess)

    def save_model(self):
        print("saveModel!")
        path = self.modelPath + self.tamer_name +"_"+str(self.sess.run(self.end_points["global_step"])) + "/"
        make_dir(path)
        path_bak = path
        path = path + self.tamer_name
        self.end_points['saver'].save(self.sess, path)

        if self.sess.run(self.end_points["global_step"]) % self.DUMP_ITR == 0:
            shutil.copytree(path_bak,self.modelPath + self.tamer_name +"_"+str(self.sess.run(self.end_points["global_step"])) + "_bak/")

    def load_projection_matrix(self,data):
        self.end_points['projection_matrix_origin'].load(data,self.sess)

    def validate(self,val_data):
        summary,global_step = self.sess.run(
            [
                self.end_points["val_summary"],
                self.end_points["global_step"]#,
                # self.end_points["l2_loss"]
            ],feed_dict={
                self.end_points["input_params"]:val_data[0],
                self.end_points["input_positions"]:val_data[1],
                self.end_points["istraining"]:False,
                self.end_points["input_rotate_angles"]:val_data[2]
            }
        )
        # print("[L2LOSS] step{} :".format(global_step),l2_loss)
        self.end_points["writer"].add_summary(summary,global_step)

    def load_use_assign(self,path):
        allVariables = tf.global_variables()
        reader = tf.train.NewCheckpointReader(path)
        for a_var in allVariables:
            itsname = a_var.name[:-2]
            # print(itsname)
            if 'patterns' in itsname:
                print("skip:",itsname)
                continue
            data = reader.get_tensor(itsname)
            a_var.load(data,self.sess)
            print("assign "+itsname+"...")    
        print(len(allVariables))

    def check_quality(self,val_data):
        results = self.sess.run([
            self.end_points["ground_truth_lumitexels_direct"],
            self.end_points["guessed_lumitexels_pd"],
            self.end_points["ground_truth_slice_pd"],
            self.end_points["guessed_lumitexels_ps"],
            self.end_points["ground_truth_slice_ps"],
            self.end_points["ground_truth_slice_length"],
            self.end_points["ps_predicted"],
            self.end_points["light_dir"],
            self.end_points["view_dir"],
            self.end_points["normal"],
            self.end_points["pos"],#12



            self.end_points['projection_matrix']
            # self.end_points["n_local_gt"],
            # self.end_points["guessed_normal"],
            # self.end_points["pd_gt"],
            # self.end_points["guessed_pd"],
        ],feed_dict={
            self.end_points["input_params"]:val_data[0],
            self.end_points["input_positions"]:val_data[1],
            self.end_points["istraining"]:False,
            self.end_points["input_rotate_angles"]:val_data[2]
        })

        _,img_height,img_width,_ = self.check_images.shape.as_list()
        img_width = img_width//5
        check_img_buffer = np.zeros([self.batch_size*5,img_height,img_width],np.float32)
        for idx,a_gt_lumi in enumerate(results[0]):

            # pos = results[12][idx]
            # view_dir = results[10][idx]  #[1,3]
            # light_dir = results[9][idx] #[lumi_size, 3]
            # n = results[11][idx]


            tmp = visualize_new(a_gt_lumi.reshape([-1]),scalerf=255.0)
            check_img_buffer[idx*5+0] = tmp
            check_img_buffer[idx*5+1] = expand_img(visualize_cube_slice(results[1][idx].reshape([-1]),sample_num=8,scalerf=255.0),8,"copy_only")
            check_img_buffer[idx*5+2] = expand_img(visualize_cube_slice(results[2][idx].reshape([-1]),sample_num=8,scalerf=255.0),8,"copy_only")

            a = results[4][idx].reshape([-1])
            tmp = visualize_cube_slice(a/a.max(),sample_num=64,scalerf=2550.0)
            cv2.putText(tmp, '{:.2f}'.format(results[5][idx][0]), (128,64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
            check_img_buffer[idx*5+4] = tmp

            a = results[3][idx].reshape([-1])/a.max()
            tmp = visualize_cube_slice(a,sample_num=64,scalerf=2550.0)
            cv2.putText(tmp, '{:.2f}'.format(results[6][idx][0]), (128,64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
            check_img_buffer[idx*5+3] = tmp
            

        
        check_img_buffer = np.transpose(check_img_buffer,axes=[1,0,2])
        check_img_buffer = np.reshape(check_img_buffer,[check_img_buffer.shape[0],self.batch_size,-1])
        check_img_buffer = np.transpose(check_img_buffer,axes=[1,0,2])
        check_img_buffer = np.expand_dims(check_img_buffer,axis=-1)
        check_img_buffer = np.minimum(check_img_buffer,255.0)
        check_img_buffer = np.maximum(check_img_buffer,0.0)

        self.check_images.load(check_img_buffer.astype(np.uint8),self.sess)
        # self.check_formfactor.load(check_img_ff_buffer.astype(np.uint8),self.sess)

        W = results[-1]
        W = W.T
        pattern_buffer = np.zeros([self.measurements_length,img_height,img_width,3],np.float32)
        for i in range(W.shape[0]):
            one_pattern = visualize_new(W[i]/W[i].max(),scalerf=255)#cook10240(W[i])#visualize(W[i]/W[i].max(),scalerf=255)
            one_pattern = np.expand_dims(one_pattern,axis = -1)
            one_pattern_pos = np.maximum(one_pattern,0.0)
            one_pattern_neg = np.minimum(one_pattern,0.0)*-1
            pattern_buffer[i] = np.concatenate([one_pattern_neg,np.zeros(one_pattern.shape),one_pattern_pos],axis = -1)
        

        pattern_buffer = np.minimum(pattern_buffer,255.0)
        pattern_buffer = np.maximum(pattern_buffer,0.0)

        self.pattern_check.load(pattern_buffer.astype(np.uint8),self.sess)

        summary,global_step = self.sess.run([
            self.end_points["check_summary"],
            self.end_points["global_step"]
        ])
        self.end_points["writer"].add_summary(summary,global_step)



        return results

       
