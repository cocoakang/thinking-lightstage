import numpy as np
import math
import os
import sys
sys.path.append("../utils/")
from dir_folder_and_files import make_dir

param_dim = 7+3+1
epsilon = 1e-3
param_bounds={}
param_bounds["n"] = (epsilon,1.0-epsilon)
param_bounds["theta"] = (0.0,math.pi)
param_bounds["a"] = (0.006,0.503)
param_bounds["pd"] = (0.0,1.0)
param_bounds["ps"] = (0.0,10.0)
param_bounds["box"] = (-50.0,50.0)
param_bounds["angle"] = (0.0,2.0*math.pi)

if __name__ == "__main__":
    data_root = sys.argv[1]
    sample_num = int(sys.argv[2])
    train_data_ratio = float(sys.argv[3])
    fixed_angle_num = int(sys.argv[4])
    make_dir(data_root)

    os.system("\"auxiliary\\exe_utils\\random_number_generator.exe\" {} {} {}".format(sample_num,param_dim,data_root+"raw_params.bin"))
    
    raw_random_numbers = np.fromfile(data_root+"raw_params.bin",np.float32).reshape([sample_num,param_dim])

    positions = raw_random_numbers[:,:3]*(param_bounds["box"][1]-param_bounds["box"][0])+param_bounds["box"][0]
    
    ns = raw_random_numbers[:,3:5]
    
    ts = raw_random_numbers[:,[5]]*(param_bounds["theta"][1]-param_bounds["theta"][0])+param_bounds["theta"][0]
    
    x_min = np.log(param_bounds["a"][0])
    x_max = np.log(param_bounds["a"][1])
    range_ax = x_max-x_min
    ax_log = raw_random_numbers[:,[6]]*range_ax+x_min
    ax_log_0_8 = np.log(np.clip(np.exp(ax_log)*0.125,param_bounds["a"][0],param_bounds["a"][1]))

    range_ay = ax_log - ax_log_0_8
    ay_log = raw_random_numbers[:,[7]]
    ay_log = ay_log*range_ay+ax_log_0_8
    
    ax = np.exp(ax_log)
    ay = np.exp(ay_log)
    # ay=ax

    pds = raw_random_numbers[:,[8]]*(param_bounds["pd"][1]-param_bounds["pd"][0])+param_bounds["pd"][0]
    pss = raw_random_numbers[:,[9]]*(param_bounds["ps"][1]-param_bounds["ps"][0])+param_bounds["ps"][0]

    rrn_for_angles = raw_random_numbers[:,[10]]
    rrn_fiexed = np.linspace(0.0,1.0,num=fixed_angle_num+1)
    rrn_idxes = np.digitize(rrn_for_angles,rrn_fiexed)
    rrn_for_angles = rrn_fiexed[rrn_idxes]

    rotate_angles = rrn_for_angles*(param_bounds["angle"][1]-param_bounds["angle"][0])+param_bounds["angle"][0]

    cooked_params = np.concatenate([positions,ns,ts,ax,ay,pds,pss,rotate_angles],axis=-1)
    np.random.shuffle(cooked_params)


    total_size = cooked_params.shape[0]
    train_data_size = int(total_size*train_data_ratio)
    val_data_size = total_size-train_data_size

    with open(data_root+"gen_log.txt","w") as plogf:
        plogf.write("total size:{}\ntrain size:{}\nvalidate size:{}".format(total_size,train_data_size,val_data_size))

    cooked_params_train = cooked_params[:train_data_size]
    cooked_params_val = cooked_params[-val_data_size:]

    cooked_params_train.astype(np.float32).tofile(data_root+"cooked_params_train.bin")
    cooked_params_val.astype(np.float32).tofile(data_root+"cooked_params_val.bin")

    np.random.shuffle(cooked_params)
    np.savetxt(data_root+"tmp.csv",cooked_params[:200],delimiter=',')