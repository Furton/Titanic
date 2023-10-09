import pandas as pd


pd.set_option("display.max_columns",200)

trainData = pd.read_csv('data/train.csv')
testData = pd.read_csv('data/test.csv')


nameMap  = {
    "Capt":       0,
    "Col":        0,
    "Major":      0,
    "Jonkheer":   1,
    "Don":        1,
    "Sir" :       1,
    "Dr":         0,
    "Rev":        0,
    "the Countess":1,
    "Dona":       1,
    "Mme":        2,
    "Mlle":       4,
    "Ms":         2,
    "Mr" :        3,
    "Mrs" :       2,
    "Miss" :      4,
    "Master" :    0,
    "Lady" :      1
}

cabinMap = {
"U":    0,
"C":    1,
"B":    2,
"D":    3,
"E":    4,
"A":    5,
"F":    6,
"G":    7,    
"T":    8
}

fare = pd.DataFrame()
fare = pd.concat([trainData.Fare, testData.Fare], ignore_index=True)
medianFare = fare.median()


age = pd.DataFrame()
age = pd.concat([trainData.Age, testData.Age], ignore_index=True)
medianAge = age.median()




def PrepareProperties(data):
    result = pd.DataFrame()
    result  = data.copy(deep=True);
    
    result.drop("Embarked", axis=1, inplace=True)
    result.drop("Ticket", axis=1, inplace=True)
    
    result.Sex = result.Sex.map({'male': 0, 'female': 1})
    
    result.Name = result.Name.apply(lambda name: name.split(',')[1].split(' ')[1][:-1])
    result.Name = result.Name.map(nameMap)
    result.Name = result.Name.fillna(result.Name.median())
    
        
    result.Cabin = result.Cabin.fillna('U')
    result.Cabin = result.Cabin.apply(lambda cabin: cabin[:1])
    result.Cabin = result.Cabin.map(cabinMap)
    
    result.Fare = result.Fare.fillna(medianFare)
    
    
    result.Age = result.Age.fillna(medianAge)


    return result;



train = pd.DataFrame()
train = PrepareProperties(trainData);
test= pd.DataFrame()
test = PrepareProperties(testData);

print(train.head())


train.to_csv('trainPure.csv')
test.to_csv('testPure.csv')



 


