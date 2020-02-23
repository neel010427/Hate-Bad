import TweetModel as tm
import pandas as pd
import numpy as np
import tensorflow as tf
from pymongo import MongoClient

def init_models():
    client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
    db = client.hatebad #database

    test = pd.DataFrame(db.hateTestSet.find())
    test = test.astype({'text': 'U', 'label': 'U', 'user': 'U'})
    test = test.drop(labels=['_id','user'], axis='columns')
    test['score'] = test['label'].map(lambda x: 1 if x == 'hate' else 0)

    training = pd.DataFrame(db.hateTrainingSet.find()) #table
    training = training.astype({'text': 'U', 'label': 'U', 'user': 'U'})
    training = training.drop(labels=['_id','user'], axis='columns')
    training['score'] = training['label'].map(lambda x: 1 if x == 'hate' else 0)

    print(test.head())
    thing = test['text']
    test_model1 = tm.Model(data=training, test_data=test)
    test_model1.test_model()
    output = test_model1.predict(test)
    print(output)


    # test = pd.DataFrame(db.positiveTestSet.find())
    # test = test.astype({'text': 'U', 'label': 'U'})
    # test = test.drop(labels=['_id'], axis='columns')
    # test['score'] = test['label'].map(lambda x: 1 if x == 'hate' else 0)

    # training = pd.DataFrame(db.positiveTrainingSet.find()) #table
    # training = training.astype({'text': 'U', 'label': 'U'})
    # training = training.drop(labels=['_id'], axis='columns')
    # training['score'] = training['label'].map(lambda x: 1 if x == 'hate' else 0)

    test_model2 = tm.Model(data=training, test_data=test)
    test_model2.test_model()

    return test_model1, test_model2

bad, good = init_models()
