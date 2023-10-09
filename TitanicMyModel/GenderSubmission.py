import pandas as pd
from sklearn.metrics import mean_squared_error


pd.set_option("display.max_columns",200)

train = pd.read_csv('trainPure.csv',index_col=0)
test = pd.read_csv('testPure.csv')


def GenderModel(test):
    return test.Sex

def Error(trainTemp, method):
    return mean_squared_error(trainTemp.Survived,GenderModel(trainTemp)) 
    

submission = pd.DataFrame()
submission['PassengerId'] = test.PassengerId
submission['Survived'] = GenderModel(test)


print(Error(train, GenderModel))

submission.to_csv('Submission/gender_submission.csv',index= False)


