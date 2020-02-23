import TweetModel as tm
import pandas as pd
import numpy as np
from pymongo import MongoClient

client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
db = client.hatebad #database

training_df = pd.DataFrame(db.hateTrainingSet.find())
training_df['label'] = training_df['label'].map(lambda x: 1 if x == 'hate' else 0)
test_df = pd.DataFrame(db.hateTestSet.find())
test_df['label'] = test_df['label'].map(lambda x: 1 if x == 'hate' else 0)

model = tm.TweetModel(name='hate', training_df=training_df, test_df=test_df)
model.train_model(num_epoch=30)
model.test_model()
model.save_model()


training_df = pd.DataFrame(db.positiveTrainingSet.aggregate([{'$sample': {'size':10000}}]))
training_df['label'] = training_df['score']
test_df = pd.DataFrame(db.positiveTestSet.aggregate([{'$sample': {'size':10000}}]))
test_df['label'] = test_df['score']

model = tm.TweetModel(name='positive', training_df=training_df, test_df=test_df)
model.train_model(num_epoch=25, num_batch=32)
model.test_model()
model.save_model()