import numpy as np
import math

PARAM_BOUNDS=(
                (0.0,1.0),#n1
                (0.0,1.0),#n2
                (0.0,2*math.pi),#theta
                (0.006,0.503),#ax
                (0.006,0.503),#ay
                (0.0,1.0),#pd
                (0.0,10.0),#ps
                )

if __name__ == "__main__":
    data_root = "test_data/"
    num_per_block = 1000
    block_num = 10

    for which_block in range(block_num):
      n1 = np.random.uniform(PARAM_BOUNDS[0][0],PARAM_BOUNDS[0][1],size=num_per_block).reshape([-1,1])
      n2 = np.random.uniform(PARAM_BOUNDS[1][0],PARAM_BOUNDS[1][1],size=num_per_block).reshape([-1,1])
      theta = np.random.uniform(PARAM_BOUNDS[2][0],PARAM_BOUNDS[2][1],size=num_per_block).reshape([-1,1])
      ax = np.exp(np.random.uniform(np.log(PARAM_BOUNDS[3][0]),np.log(PARAM_BOUNDS[2][1]),size=num_per_block).reshape([-1,1]))
      ay = np.exp(np.random.uniform(np.log(PARAM_BOUNDS[4][0]),np.log(PARAM_BOUNDS[4][1]),size=num_per_block).reshape([-1,1]))
      pd = np.random.uniform(PARAM_BOUNDS[5][0],PARAM_BOUNDS[5][1],size=num_per_block).reshape([-1,1])
      ps = np.random.uniform(PARAM_BOUNDS[6][0],PARAM_BOUNDS[6][1],size=num_per_block).reshape([-1,1])
      res = np.concatenate([n1,n2,theta,ax,ay,pd,ps],axis=-1)
      res.astype(np.float32).tofile(data_root+"test_param{}.bin".format(which_block))