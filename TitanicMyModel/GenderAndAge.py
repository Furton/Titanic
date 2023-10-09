import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error


pd.set_option("display.max_columns",200)

train = pd.read_csv('trainPure.csv',index_col=0)
test = pd.read_csv('testPure.csv')





a = [1,1,0]
delta = 0.05

def RandomRange():
    return 1 - 2*random.random()

def UnitySeries(data):
   return pd.Series(np.full(len(data), 1))

def GenderAgeModel(data):
      unitySeries = UnitySeries(data)
      max = data.Age.max()
      result = pd.Series()
      result = (a[1]*data.Sex + a[2]*(data.Age/max) >= a[0]*unitySeries).map({False: 0,True: 1}).copy(deep=True);
      return result



    
    
def Correct(trainTemp, method):
    return 1 - mean_squared_error(trainTemp.Survived,GenderAgeModel(trainTemp)) 
    




submission = pd.DataFrame()
submission['PassengerId'] = test.PassengerId
submission['Survived'] = GenderAgeModel(test)


correctPercentage = Correct(train, GenderAgeModel)
print(correctPercentage)


a = [a[0] + RandomRange(),a[1] + RandomRange(),a[2] + RandomRange()]
correctPercentage = Correct(train, GenderAgeModel)
print(correctPercentage)

submission.to_csv('Submission/gender_and_age_submission.csv',index= False)