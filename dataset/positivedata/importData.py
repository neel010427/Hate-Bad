import pandas as pd
import numpy as np
from pymongo import MongoClient

data = pd.read_csv('data.csv',
  names = ('score', 'a', 'b', 'c', 'd', 'text'), usecols=['score', 'text'], index_col = False, encoding = "ISO-8859-1")
data = data[['text', 'score']]
data['score'] = data['score'].apply(lambda x: 0 if x < 3 else 1)
#print(data.head())
mask = np.random.choice(len(data), size=int(len(data) * 0.8), replace=False)
training_set = data.iloc[mask, :]
test_set = data.iloc[-mask, :]

print('imported data')

dataTest = test_set.to_dict('records')
dataTrain = training_set.to_dict('records')

print('data to dictionary')

client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.hatebad
testTable = db.positiveTestSet
trainTable = db.positiveTrainingSet

testTable.drop()
trainTable.drop()

print('inserting into test')
testTable.insert_many(dataTest)
print('inserting into train')
trainTable.insert_many(dataTrain)
