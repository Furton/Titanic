import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error
import trainModels


pd.set_option("display.max_columns",200)

train = pd.read_csv('trainPure.csv',index_col=0)
test = pd.read_csv('testPure.csv')


strId = 'Submission/linear_submission.csv' 
trainModels.Train(trainModels.LinearModel,strId,0.04,np.array([0,0,0,0,0,0]), train,test);

  