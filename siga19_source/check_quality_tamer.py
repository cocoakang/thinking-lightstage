import tensorflow as tf
import sys
sys.path.append('../utils/')
import math
import numpy as np
import cv2
from dir_folder_and_files import make_dir
from lumitexel_related import visualize_new
sys.path.append('./auxiliary/tf_ggx_render_newparam')
from tf_ggx_render import tf_ggx_render

class Tame_Tamer:
    def __init__(self,train_configs,for_train=True):
        self.end_points = {}
        if for_train:
            self.parameter_len = train_configs["parameter_len"]
            self.lumitexel_length = train_configs["lumitexel_length"]
            self.measurements_length = train_configs["measurements_length"]
            self.learning_rate = train_configs["learning_rate"]
            self.tamer_name = train_configs["tamer_name"]
            self.logPath = train_configs["log_details_dir"]
            make_dir(self.logPath)
            self.modelPath = train_configs["logPath"]+"models/"
            make_dir(self.modelPath)
            self.batch_size = train_configs["batch_size"]

            standard_rendering_parameters = {}

            standard_rendering_parameters["parameter_len"] = self.parameter_len
            standard_rendering_parameters["batch_size"] = self.batch_size
            standard_rendering_parameters["lumitexel_size"] = self.lumitexel_length
            standard_rendering_parameters["is_grey_scale"] = True
            standard_rendering_parameters["config_dir"] = "../tf_ggx_render/tf_ggx_render_configs_1x1/"
            self.end_points["standard_rendering_parameters"] = standard_rendering_parameters

            self.RENDER_SCALAR = 5*1e3/math.pi
        else:
            self.parameter_len = train_configs["parameter_len"]

        
    
    def draw_param_guesser(self,measurements,scope_name,renderer,isTraining,part_trainable = True,af = tf.nn.relu):
        '''
        measurements = [batch,measure_ments]
        return =[batch,param_length]

        scope_name: a string, the name of this part
        '''
        self.ae_trainable = True
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
            block_ns = tf.variable_scope('fc_block2')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 2")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 512
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
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
                size = 1024
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            ##medium start###
            block_ns = tf.variable_scope('fc_block5')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 5")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn4",x_n,size,isTraining,af,self.ae_trainable)
                size = 2048
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block6')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 6")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            ##deep start###
            block_ns = tf.variable_scope('fc_block7')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 7")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 2048
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block8')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 8")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block9')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 9")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size = 4096
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            block_ns = tf.variable_scope('fc_block10')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw fc block 10")
                if (withBN):
                    x_n = tamerTools.batch_norm_with_somany("bn1",x_n,size,isTraining,af,self.ae_trainable)
                size=8192
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], size, activation_fn=af, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
                    x_n[idx] = tf.contrib.layers.dropout(x_n[idx],keep_prob=keep_ratio,is_training = isTraining)
            ##deep end###
            block_ns = tf.variable_scope('deep_output')#,reuse = tf.AUTO_REUSE)
            with block_ns:
                print("draw output decoder")
                for idx,value in enumerate(x_n):
                    x_n[idx] = tf.contrib.layers.fully_connected(x_n[idx], self.lumitexel_length, activation_fn=None, trainable=self.ae_trainable,scope=tf.contrib.framework.get_name_scope(),reuse = tf.AUTO_REUSE)
        return x_n[0]

    def draw_train_net(self,net_name,projectionMatrix_trainable):
        '''
        net_name: a string,the name of this net
        '''
        with tf.variable_scope(net_name,reuse=tf.AUTO_REUSE):
            #[1]define input parameters
            isTraining = tf.placeholder(tf.bool,name = "isTraining_ph")
            self.end_points["istraining"] = isTraining

            #[2]draw rendering net,render ground truth 
            ground_truth_lumitexels = tf.placeholder(tf.float32,shape=[self.batch_size,self.lumitexel_length],name="ground_truth_lumitexel")
            self.end_points["ground_truth_lumitexels"] = ground_truth_lumitexels

            #[3]draw linear projection layer
            with tf.variable_scope('linear_projection',reuse = tf.AUTO_REUSE):
                kernel = tf.get_variable(name = 'projection_matrix',
                                        trainable=projectionMatrix_trainable,
                                        shape = [self.lumitexel_length,self.measurements_length],
                                        initializer=tf.contrib.layers.xavier_initializer()
                                        )#[lightnum,measure_ments]
                self.end_points['projection_matrix'] = kernel

                measure_ments = tf.matmul(ground_truth_lumitexels,kernel)#[batch,measure_ments]
                self.end_points["measure_ments"] = measure_ments
            
            #[3.1]noiser
            with tf.variable_scope("noiser"):
                noise = tf.random_normal(tf.shape(measure_ments),mean = 0.0,stddev = 0.01,name="noise_generator")+1.
                measure_ments_noised = tf.multiply(measure_ments,noise)
            #[4]draw guesser net
            guessed_lumitexels = self.draw_param_guesser(measure_ments_noised,"param_guesser",None,isTraining)#[batch,param_length]
            guessed_lumitexels = tf.expand_dims(guessed_lumitexels,axis=-1)
            self.end_points["guessed_lumitexels"] = guessed_lumitexels

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
            
        
        #[8.2] initializer
        init = tf.global_variables_initializer()
        self.end_points['init'] = init
        #[8.3] saver
        saver = tf.train.Saver()
        self.end_points['saver'] = saver
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        self.sess = tf.Session(config=config)


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
                self.end_points["istraining"]:True
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

    def restore_model_assign(self,path):
        donnotList = []
        allVariables = tf.global_variables()
        reader = tf.train.NewCheckpointReader(path)
        un_assigned_param = []
        for a_var in allVariables:
            itsname = a_var.name[:-2]
            shouldContinue = False
            for notwanted in donnotList:
                if notwanted in itsname:
                    shouldContinue = True
                    break
            if shouldContinue:
                # print("skip:",itsname)
                un_assigned_param.append(itsname)
                continue
            data = reader.get_tensor(itsname)
            a_var.load(data,self.sess)
            print("assign "+itsname+"...")   
        print("----------------------------")
        print("These were skipped. CAUTION!")
        for aname in un_assigned_param:
            print(aname)

    def save_model(self):
        print("saveModel!")
        path = self.modelPath + self.tamer_name +"_"+str(self.sess.run(self.end_points["global_step"])) + "\\"
        make_dir(path)
        path = path + self.tamer_name
        self.end_points['saver'].save(self.sess, path)
    
    def load_projection_matrix(self,data):
        self.end_points['projection_matrix'].load(data,self.sess)

    def validate(self,val_data):
        summary,global_step = self.sess.run(
            [
                self.end_points["val_summary"],
                self.end_points["global_step"]#,
                # self.end_points["l2_loss"]
            ],feed_dict={
                self.end_points["input_params"]:val_data[0],
                self.end_points["input_positions"]:val_data[1],
                self.end_points["istraining"]:False
            }
        )
        # print("[L2LOSS] step{} :".format(global_step),l2_loss)
        self.end_points["writer"].add_summary(summary,global_step)

    def check_quality(self,val_data):
        results = self.sess.run([
            self.end_points["ground_truth_lumitexels"],
            self.end_points["guessed_lumitexels"],
            self.end_points["measure_ments"]
        ],feed_dict={
            self.end_points["ground_truth_lumitexels"]:val_data[0],
            self.end_points["istraining"]:False
        })
        return results

       
