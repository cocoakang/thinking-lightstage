from tf_ggx_render import tf_ggx_render
import sys
import tensorflow as tf
import numpy as np
import math
import scipy
import cv2
import time
from tf_ggx_render_utils import visualize_init,visualize_new,make_dir
import scipy.optimize as opt

MAX_TRY_ITR = 5
PARAM_BOUNDS=(
                (0.0,1.0),#n1
                (0.0,1.0),#n2
                (0.0,2*math.pi),#theta
                (0.006,0.503),#ax
                (0.006,0.503),#ay
                (0.0,1.0),#pd
                (0.0,10.0),#ps
                )
RENDER_SCALAR =  5*1e3/math.pi
FITTING_SCALAR = 1.0/RENDER_SCALAR

def __compute_init_pdps(for_what,be_fitted,standard_lumi):
    '''
    for_what = "pd" or "ps"
    be_fitted = [batch,light_num]. MEANING: the data to be fitted
    '''

    #[step2] rendering
    rendered_lumitexel = np.squeeze(standard_lumi,axis=-1)#[batch,light_num]
    be_fitted_idxs = np.argsort(be_fitted)
    if for_what == "pd":
        psd_idxs = be_fitted_idxs#be_fitted_idxs[:,:self.lumitexel_size//2]
    elif for_what == "ps":
        psd_idxs = be_fitted_idxs[:,self.lumitexel_size//2:]
    else:
        print("[ERROR]unsupported init")
        exit()

    x_axis_index=np.tile(np.arange(len(rendered_lumitexel)), (psd_idxs.shape[1],1)).transpose()
    part_lumitexels_befitted = be_fitted[x_axis_index,psd_idxs]
    part_lumitexels_rendered = rendered_lumitexel[x_axis_index,psd_idxs]
    
    downside = np.einsum('ij, ij->i', part_lumitexels_rendered, part_lumitexels_rendered)
    upperside = np.einsum('ij, ij->i', part_lumitexels_rendered, part_lumitexels_befitted)
    scalers = upperside/downside
    return scalers

def __fitting_pdps_from_grey(one_of_rgb_lumi,standard_lumi_d,standard_lumi_s,pdps):
    '''
    one_of_rgb_lumi = [batch,light_num]
    standard_lumi_d = [batch,light_num,1]
    standard_lumi_s = [batch,light_num,1]
    pdps = [batch,2,1]
    '''
    b = tf.expand_dims(one_of_rgb_lumi,axis=-1)#[batch,light_num,1]
    a = tf.concat([standard_lumi_d,standard_lumi_s],axis=-1)#[batch,light_num,2]
    at = tf.transpose(a,perm=[0,2,1])#[batch,2,light_num]
    atb = tf.matmul(at,b)#[batch,2,1]
    ata = tf.matmul(at,a)#[batch,2,2]
    ata_pdps = tf.matmul(ata,pdps)#[batch,2,1]

    pdps_loss = tf.nn.l2_loss(tf.squeeze(ata_pdps-atb,axis=2))#[1]

    return pdps_loss


if __name__ == "__main__":
    thread_id = int(sys.argv[1])
    data_path = sys.argv[2]
    log_path = data_path+"logs_thread{}/".format(thread_id)#sys.argv[3]
    make_dir(log_path)
    log_start_idx = 0#thread_id*int(sys.argv[4])
    origin_colorful = int(sys.argv[3]) == 1
    need_dump = int(sys.argv[4]) == 1
    data_file_name_base = sys.argv[5]
    if_use_guessed_param = int(sys.argv[6]) == 1

    data_file_name = data_file_name_base+"{}.bin".format(thread_id)

    parameters = {}
    parameters["shrink_size"] = 2
    parameters["batch_size"] = 1
    parameters["lumitexel_size"] = 24576//parameters["shrink_size"]//parameters["shrink_size"]
    parameters["is_grey_scale"] = True
    parameters["parameter_len"] = 7
    parameters["config_dir"] = './tf_ggx_render_configs_{}x{}/'.format(parameters["shrink_size"],parameters["shrink_size"])#指向随包的config文件夹

    visualize_init(parameters["config_dir"]+"visualize_idxs.bin",parameters["lumitexel_size"])

    

    renderer = tf_ggx_render(parameters)#实例化渲染器

    # test_params = np.fromfile(data_path+"test_param{}.bin".format(thread_id),np.float32).reshape([-1,parameters["parameter_len"]])
    wanted_lights_idx = np.fromfile(parameters["config_dir"]+"light_wanted.bin",np.int32)#.reshape([1,-1]),[fitting_data.shape[0],1])#[-1,light_num]
    
    fitting_data = np.fromfile(data_path+data_file_name,np.float32).reshape([-1,wanted_lights_idx.shape[0]])*FITTING_SCALAR#[-1,24576]

    if origin_colorful:
        fitting_data_colorful = fitting_data.copy().reshape([-1,3,parameters["lumitexel_size"]])#[totalsize,3,lumitexel_len]
        fitting_data = np.mean(fitting_data_colorful,axis=1)#[totalsize,lumitexel_len]
    
    if if_use_guessed_param:
        guessed_param = np.fromfile(data_path+"guessed_params.bin",np.float32).reshape([-1,7])
        assert guessed_param.shape[0] == fitting_data.shape[0]

    if origin_colorful:
        RESULT_params = np.zeros([fitting_data.shape[0],11],np.float32)
    else:
        RESULT_params = np.zeros([fitting_data.shape[0],7],np.float32)
    

    # for a_colorful_lumi in fitting_data_colorful:
    #     origin_img_3ch = []
    #     for a_lumi in a_colorful_lumi:
    #         origin_img = visualize_new(np.reshape(a_lumi,[-1]),scalerf=1.0,len=64//parameters["shrink_size"])
    #         origin_img_3ch.append(origin_img)
    #     origin_img_3ch = np.asarray(origin_img_3ch,np.float32)
    #     origin_img_3ch = np.transpose(origin_img_3ch,axes=[1,2,0])[:,:,::-1]
    #     cv2.imshow("img_o",origin_img_3ch)
    #     cv2.waitKey(0)
    # exit(0)


    # print(test_params.shape)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        ##########################################
        ####[STEP 1] draw rendering graph
        ##########################################
        #define inputs
        input_params = tf.get_variable(name = "input_parameters",dtype=tf.float32,shape = [parameters["batch_size"],parameters["parameter_len"]])
        #tf.placeholder(tf.float32,shape=[parameters["batch_size"]*parameters["parameter_len"]],name="render_params")
        # input_params_reshaped = tf.reshape(input_params,shape=[parameters["batch_size"],parameters["parameter_len"]])
        input_positions = tf.get_variable(name="position",dtype=tf.float32,shape=[parameters["batch_size"],3],trainable=False)
        #input_positions = tf.placeholder(tf.float32,shape=[parameters["batch_size"],3],name="render_positions")
        input_labels = tf.get_variable(name = "input_labels" ,dtype=tf.float32, shape = [parameters["batch_size"],parameters["lumitexel_size"]],trainable=False)
        # rendering_scaler = tf.constant(1.0)#RENDER_SCALAR*FITTING_SCALAR)#TODO CAN BE SET

        #draw rendering net
        rendered_res = renderer.draw_rendering_net(input_params,input_positions,"my_little_render",with_cos=False)
        n_dot_view_dir,n_dot_view_penalty = renderer.calculate_n_dot_view("my_little_render")
        init_params_ps0,init_params_pd0 = renderer.param_initializer(input_labels,"my_little_render")

        if parameters["is_grey_scale"]:
            l2_loss = tf.nn.l2_loss(tf.reshape(input_labels,[parameters["batch_size"],parameters["lumitexel_size"],1])-rendered_res)
        else:
            l2_loss = tf.nn.l2_loss(tf.reshape(input_labels,[parameters["batch_size"],parameters["lumitexel_size"],3])-rendered_res)

        total_loss = l2_loss#+n_dot_view_penalty
        from tf_ggx_render_utils import print_loss,print_step,init_step,report_step
        epsilon = 1e-6
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            total_loss,
            # options={'maxfun':9999999999,'maxiter': 100,"maxls":20},
            options={'maxiter': 100},
            var_to_bounds={input_params: ([epsilon,epsilon,0.0,0.006,0.006,0.0,0.0], [1.0-epsilon,1.0-epsilon,2*math.pi,0.503,0.503,1.0,10.0])},
            method='L-BFGS-B'
        )
        # method='TNC')#,epsilon=1.0)#

        ##########################################
        ####[STEP 1.5] draw fitting pdps net
        ##########################################
        if origin_colorful:
            one_of_rgb_lumi = tf.get_variable(name = "one_of_rgb_lumi" ,dtype=tf.float32, shape = [parameters["batch_size"],parameters["lumitexel_size"]],trainable=False)
            standard_lumi_d = tf.get_variable(name = "standard_lumi_d" ,dtype=tf.float32, shape = [parameters["batch_size"],parameters["lumitexel_size"],1],trainable=False)
            standard_lumi_s = tf.get_variable(name = "standard_lumi_s" ,dtype=tf.float32, shape = [parameters["batch_size"],parameters["lumitexel_size"],1],trainable=False)
            render_standard_lumi_d = tf.assign(standard_lumi_d,rendered_res,name="assign_render_standard_lumi_d")
            render_standard_lumi_s = tf.assign(standard_lumi_s,rendered_res,name="assign_render_standard_lumi_d")
            fitted_pdps = tf.get_variable(name="fitted_pdps",dtype=tf.float32,shape=[parameters["batch_size"],2,1])
            fitting_pdps_loss = __fitting_pdps_from_grey(one_of_rgb_lumi,standard_lumi_d,standard_lumi_s,fitted_pdps)
            optimizer_pdps = tf.contrib.opt.ScipyOptimizerInterface(
                fitting_pdps_loss,
                var_list=[fitted_pdps],
                options={'maxiter': 100},
                var_to_bounds={input_params: ([epsilon,epsilon,0.0,0.006,0.006,0.0,0.0], [1.0-epsilon,1.0-epsilon,2*math.pi,0.503,0.503,1.0,10.0])},
                method='L-BFGS-B')



        ##########################################
        ####[STEP 2] init constants
        ##########################################
        #global variable init should be called before rendering
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # rendering_fun = lambda x: sess.run(l2_loss,feed_dict={
        #     input_params:x
        # })
        ##########################################
        #####[STEP 2]fitting here
        ##########################################
        # fparam = open(data_path+"fitted_params{}.bin".format(thread_id),"wb")
        plogf = open(log_path+"logs.txt","w")
        # step_costs = []
        fitting_time_cost_preparation = 0
        fitting_time_cost_nta = 0
        fitting_time_cost_pdps = 0
        tmp_from = 0
        total_time_cost= 0
        # pfittedlumitexelf = open(log_path+"fitted.bin","wb")
        for idx,a_lumi in enumerate(fitting_data[tmp_from:]):
            if idx % 100 == 0:
                print("[THREAD{}]{}/{} cost time:{}s".format(thread_id,idx,fitting_data.shape[0],total_time_cost))
            ##########################################
            ####[STEP 2.1] compute init
            ##########################################
            #load labels an position
            START_TIME = time.time()
            input_positions.load(np.zeros([parameters["batch_size"],3],np.float32),sess)#TODO CAN BE SET
            r_2_cos = np.squeeze(renderer.get_r_2_cos(sess,"my_little_render"),axis=2)#[batch,lightnum]
            a_lumi_expanded = np.expand_dims(a_lumi,axis=0)#[1,lightnum]
            a_lumi_de_formfactor = a_lumi_expanded*r_2_cos#[1,lightnum]
            input_labels.load(a_lumi_de_formfactor,sess)#a_lumi=[24576] ->[1,24576]
            
            if origin_colorful:
                a_lumi_rgbs = fitting_data_colorful[tmp_from+idx]#[3,lightnum]
                a_lumi_rgbs_de_formfactor = a_lumi_rgbs*r_2_cos

            #[INITIALIZE]calculate initial params and load data
            if if_use_guessed_param:
                x0 = guessed_param[[tmp_from+idx]]
                standard_params = guessed_param[[tmp_from+idx]]
            else:
                standard_params = sess.run(init_params_ps0)
                input_params.load(standard_params,sess)
                standard_lumi = sess.run(rendered_res)

                pds = __compute_init_pdps("pd",a_lumi_de_formfactor,standard_lumi)
                
                standard_params[:,5] = pds*0.5#np.expand_dims(pds,1)#[batch,1]
                standard_params[:,6] = pds*0.5#np.zeros_like(pss,np.float32)#np.expand_dims(pss,1)#[batch,1]
                
                x0 = standard_params
            input_params.load(x0,sess)
            fitting_time_cost_preparation +=time.time() - START_TIME

            ##########################################
            ####[STEP 2.2] fitting here
            ##########################################
            #play here
            # print("training...") 
            START_TIME = time.time()

            for try_itr in range(MAX_TRY_ITR):
                init_step()
                optimizer.minimize(
                    sess,
                    step_callback=print_step,
                    # loss_callback=print_loss,
                    # fetches=[total_loss, input_params]
                )
                if report_step() == 1:
                    print("[ERROR]recompute!#######################")
                    x0 = standard_params*np.random.normal(1.0,0.01,size=standard_params.shape)
                    input_params.load(x0,sess)
                else:
                    break
            fitting_time_cost_nta +=time.time() - START_TIME
            # print("done.")
            fitted_params_grey = sess.run(input_params)#[batch,7]
            RESULT_params[tmp_from+idx][:5] = fitted_params_grey.reshape([-1])[:5]
            ##########################################
            ####[STEP 2.3] fitting rgb
            ##########################################
            START_TIME = time.time()
            if origin_colorful:
                tmp_params = fitted_params_grey.copy()
                tmp_params[:,5] = 0
                tmp_params[:,6] = 1.0
                input_params.load(tmp_params,sess)
                sess.run(render_standard_lumi_s)#load standard ps lumi
                tmp_params[:,5] = 1.0
                tmp_params[:,6] = 0.0
                input_params.load(tmp_params,sess)
                sess.run(render_standard_lumi_d)#load standard pd lumi
                # print("training pdps...")
                for which_channel in range(3):
                    one_of_rgb_lumi.load(a_lumi_rgbs_de_formfactor[which_channel].reshape([1,-1]),sess)
                    optimizer_pdps.minimize(
                        sess#,
                        # step_callback=print_step,
                        # loss_callback=print_loss,
                        # fetches=[fitting_pdps_loss, fitted_pdps]
                    )
                    fitted_params_pdps = sess.run(fitted_pdps)
                    RESULT_params[tmp_from+idx][5+which_channel] = fitted_params_pdps[0][0]
                    RESULT_params[tmp_from+idx][8+which_channel] = fitted_params_pdps[0][1]
                # print("done.")

            fitting_time_cost_pdps +=time.time() - START_TIME
            ##########################################
            ####[STEP 2.3] check result
            ##########################################
            if need_dump or idx % 100 == 0:
                if origin_colorful:
                    tmp_params = np.zeros([7],np.float32)
                    fitted_img_3ch = []
                    origin_img_3ch = []
                    for which_channel in range(3):
                        tmp_params[:5] = RESULT_params[tmp_from+idx][:5]
                        tmp_params[5] = RESULT_params[tmp_from+idx][5+which_channel]
                        tmp_params[6] = RESULT_params[tmp_from+idx][8+which_channel]
                        input_params.load(tmp_params.reshape([1,7]),sess)
                        fitted_lumi = sess.run(rendered_res)
                        origin_img = visualize_new(np.reshape(a_lumi_rgbs_de_formfactor[which_channel],[-1]),scalerf=RENDER_SCALAR*2e-3,len=64//parameters["shrink_size"])
                        fitted_img = visualize_new(np.reshape(fitted_lumi,[-1]),scalerf=RENDER_SCALAR*2e-3,len=64//parameters["shrink_size"])
                        fitted_img_3ch.append(fitted_img)
                        origin_img_3ch.append(origin_img)
                    
                    fitted_img_3ch = np.asarray(fitted_img_3ch,np.float32)
                    origin_img_3ch = np.asarray(origin_img_3ch,np.float32)
                    fitted_img_3ch = np.transpose(fitted_img_3ch,axes=[1,2,0])[:,:,::-1]
                    origin_img_3ch = np.transpose(origin_img_3ch,axes=[1,2,0])[:,:,::-1]
                    cv2.imwrite(log_path+"img{}_o.png".format(log_start_idx+idx),origin_img_3ch)
                    cv2.imwrite(log_path+"img{}_f.png".format(log_start_idx+idx),fitted_img_3ch)
                    
                else:
                    fitted_lumi = sess.run(rendered_res)
                    # fitted_lumi.astype(np.float32).tofile(pfittedlumitexelf)
                    input_img = visualize_new(np.reshape(a_lumi_de_formfactor,[-1]),scalerf=RENDER_SCALAR*1e-3,len=64//parameters["shrink_size"])
                    fitted_img = visualize_new(np.reshape(fitted_lumi,[-1]),scalerf=RENDER_SCALAR*1e-3,len=64//parameters["shrink_size"])
                    cv2.imwrite(log_path+"img{}_o.png".format(log_start_idx+idx),input_img)
                    cv2.imwrite(log_path+"img{}_f.png".format(log_start_idx+idx),fitted_img)

            if idx % 5000 == 0:
                RESULT_params.astype(np.float32).tofile(log_path+"fitted_{}_log_{}.bin".format(thread_id,idx))
            
            total_time_cost = fitting_time_cost_preparation+fitting_time_cost_nta+fitting_time_cost_pdps
            # step_costs.append(step_)
        # pfittedlumitexelf.close()
        total_time_cost = fitting_time_cost_preparation+fitting_time_cost_nta+fitting_time_cost_pdps
        plogf.write("TIME COST(preparation):{}s\n".format(fitting_time_cost_preparation))
        plogf.write("TIME COST(nta):{}s\n".format(fitting_time_cost_nta))
        plogf.write("TIME COST(pdps):{}s\n".format(fitting_time_cost_pdps))
        plogf.write("TIME COST(total):{}s\n".format(total_time_cost))
        
        # step_costs =np.array(step_costs)
        # np.savetxt(log_path+"steps.csv",step_costs.reshape([-1,1]),delimiter=',')
        # plogf.write("AVERAGE STEP:{}\n".format(step_costs.mean()))
        plogf.close()
    RESULT_params.astype(np.float32).tofile(data_path+"fitted_{}.bin".format(thread_id))
    # print("TIME COST:{}s".format(total_time_cost))
    # fparam.close()
    # print("[THREAD {}]      OOOOOOOOOOVER".format(thread_id))