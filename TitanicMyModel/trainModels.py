import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error

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

def LinearModel(data, aTemp):
      unitySeries = UnitySeries(data)
      result = pd.Series()
      result = (aTemp[1]*data.Sex + aTemp[2]*(data.Age/data.Age.max()) + aTemp[3]*(data.Pclass/data.Pclass.max()) + aTemp[4]*(data.Fare/data.Fare.max()) + aTemp[4]*(data.Name/data.Name.max()) + aTemp[5]*(data.Cabin/data.Cabin.max()) >= aTemp[0]*unitySeries).map({False: 0,True: 1}).copy(deep=True);
      return result

    
    
def Correct(trainTemp, method,aTemp):
    return 1 - mean_squared_error(trainTemp.Survived,method(trainTemp,aTemp)) 
    






def Train(Model,modelName,delta,a,train,test):
  print(a)
  correctPercentage = Correct(train, Model,a)
  print(correctPercentage)
  i = 0
  while i<1600:
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