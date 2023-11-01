from option import Option
import numpy as np
import pandas as pd
option=Option()
class Loader :
    def __init__(self,Option:option):
        self.num_person = option.num_person
        self.num_gesture = option.num_gesture
        self.num_channel= option.num_channel
        self.num_row_perpage = option.num_row_perpage
        self.folder_path=option.folder_path

    def load_3d(self):
        num_samples=self.num_person*self.num_gesture
        num_features=self.num_channel
        num_steps=self.num_row_perpage
        x=np.zeros((num_samples,num_features,num_steps))
        y=np.zeros((num_samples))
        for i in range(1,self.num_person+1):
            for j in range(1,self.num_gesture+1):
                dftemp=pd.read_excel(self.folder_path+'/{}{}.xls'.format(i,j))
                x_index=(i-1)*self.num_gesture+j-1
                x[x_index,:,:]=dftemp.T
                y[x_index]=int(x_index+1)
        return x,y