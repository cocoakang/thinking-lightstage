import numpy as np

class Synthesis_Data_Generator:
    def __init__(self,config):
        with open(config["data_pan"],"rb") as pf:
            _ = np.fromfile(pf,np.float32,1)
            self.data = np.fromfile(pf,np.float32).reshape([-1,24576])
            # self.data = np.mean(self.data,axis=1)
        print("data num:",self.data.shape[0])
        self.batch_size = 60
        self.data_ptr = 0

    def gen_batch(self):
        tmp_data = self.data[self.data_ptr:self.data_ptr+self.batch_size]
        valids = tmp_data.shape[0]
        self.data_ptr+=valids
        if tmp_data.shape[0] == 0:
            return None,False,0
        elif tmp_data.shape[0] < self.batch_size:
            tmp_data = np.concatenate([tmp_data,np.zeros([self.batch_size-valids,24576],np.float32)],axis=0)
        return (tmp_data,),True,valids