import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error


pd.set_option("display.max_columns",200)

train = pd.read_csv('trainPure.csv',index_col=0)
test = pd.read_csv('testPure.csv')







def RandomRange(aTemp):
    return np.ones(len(aTemp)) - 2*np.random.rand(len(aTemp))

def UnitySeries(data):
   return pd.Series(np.full(len(data), 1))

def GenderAgeModel(data, aTemp):
      unitySeries = UnitySeries(data)
      max = data.Age.max()
      result = pd.Series()
      result = (aTemp[1]*data.Sex + aTemp[2]*(data.Age/max) >= aTemp[0]*unitySeries).map({False: 0,True: 1}).copy(deep=True);
      return result


def GenderAgeModelNonL(data, aTemp):
      unitySeries = UnitySeries(data)
      max = data.Age.max()
      result = pd.Series()
      result = (aTemp[1]*data.Sex + aTemp[2]*(data.Age/max) + aTemp[3]*(data.Age/max) >= aTemp[0]*unitySeries).map({False: 0,True: 1}).copy(deep=True);
      return result
    
    
def Correct(trainTemp, method,aTemp):
    return 1 - mean_squared_error(trainTemp.Survived,method(trainTemp,aTemp)) 
    






def Train(Model,modelName,delta,a):
  print(a)
  correctPercentage = Correct(train, Model,a)
  print(correctPercentage)
  i = 0
  while i<1000:
      b = np.copy(delta*RandomRange(a) + a)
      correctPercentageTemp = Correct(train, Model,b)
      if correctPercentageTemp > correctPercentage:
          correctPercentage = correctPercentageTemp            
          a = np.copy(b)
      i = i + 1  
  print(correctPercentage)
  print(a)
  submission = pd.DataFrame()
  submission['PassengerId'] = test.PassengerId
  submission['Survived'] = Model(test,a)
  submission.to_csv(modelName,index= False)
  
 

strId = 'Submission/gender_and_age_submission.csv' 
Train(GenderAgeModel,strId,0.04,np.array([0,0,0]));


strId = 'Submission/gender_and_age_submission.csv' 
Train(GenderAgeModelNonL,strId,0.04,np.array([0,0,0,0]));



  