import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error


pd.set_option("display.max_columns",200)

train = pd.read_csv('trainPure.csv',index_col=0)
test = pd.read_csv('testPure.csv')





a = np.array([1,1,0])
delta = 0.2

def RandomRange():
    return 1 - 2*random.random()

def UnitySeries(data):
   return pd.Series(np.full(len(data), 1))

def GenderAgeModel(data, aTemp):
      unitySeries = UnitySeries(data)
      max = data.Age.max()
      result = pd.Series()
      result = (aTemp[1]*data.Sex + aTemp[2]*(data.Age/max) >= aTemp[0]*unitySeries).map({False: 0,True: 1}).copy(deep=True);
      return result



    
    
def Correct(trainTemp, method,aTemp):
    return 1 - mean_squared_error(trainTemp.Survived,GenderAgeModel(trainTemp,aTemp)) 
    







correctPercentage = Correct(train, GenderAgeModel,a)
print(correctPercentage)



i = 0
while i<1000:
  b = np.copy(delta*np.array([RandomRange(),RandomRange(),RandomRange()]) + a)
  correctPercentageTemp = Correct(train, GenderAgeModel,b)
  if correctPercentageTemp > correctPercentage:
     correctPercentage = correctPercentageTemp 
     a = np.copy(b)
     print(correctPercentage)
  i = i + 1  

print(a)
submission = pd.DataFrame()
submission['PassengerId'] = test.PassengerId
submission['Survived'] = GenderAgeModel(test,a)
submission.to_csv('Submission/gender_and_age_submission.csv',index= False)