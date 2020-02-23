import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras
from pymongo import MongoClient

# training_df = pd.DataFrame(db.hateTrainingSet.find()) #table
# training_df = training_df.astype({'text': 'U', 'label': 'U', 'user': 'U'})
# training_target = np.array(training_df['label'].map(lambda x: 1 if x == 'hate' else 0))
# training_df = training_df.drop(labels=['_id','user','label'], axis='columns')

# test_df = pd.DataFrame(db.hateTestSet.find()) #table
# test_df = test_df.astype({'text': 'U', 'label': 'U', 'user': 'U'})
# test_target = np.array(test_df['label'].map(lambda x: 1 if x == 'hate' else 0))
# test_df = test_df.drop(labels=['_id','user','label'], axis='columns')

# training = tf.data.Dataset.from_tensor_slices((training_df['text'].values, training_target))
# test = tf.data.Dataset.from_tensor_slices((test_df['text'].values, test_target))

# embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                            dtype=tf.string, trainable=True)

# model = tf.keras.Sequential()
# model.add(hub_layer)
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(8, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['binary_accuracy'])

# history = model.fit(training.shuffle(len(training_df)).batch(16), epochs=25, verbose=1)

# results = model.evaluate(test.batch(16), verbose=2)

# for name, value in zip(model.metrics_names, results):
#   print("%s: %.3f" % (name, value))

# predictions = model.predict(test.batch(16))

# something = test_df.copy()
# something['prediction'] = predictions > 0.7
# something['prediction'] = something['prediction'].map(lambda x: 1 if x else 0)
# something['actual'] = test_target

# print(something[something['prediction'] > something['actual']])
# print(something[something['prediction'] < something['actual']])

# cutoff = np.log(0.75 / (1 + 0.75))
# for i in range(len(test_df)):
#   if predictions[i] <= 0.75:
#     print(test_df['text'][i])

class TweetModel:
  def __init__(self, name='name', training_df=None, test_df=None, load_file=False):
    self.name = name
    self.model = None
    self.test_df = self.training_df = self.test_target = self.training_target = 0
    self.cutoff = 0.7
    if load_file:
      self.model = tf.keras.models.load_model('saved_model\\' + name)
    else:
      self.get_data(training_df, test_df)
      self.embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

  def get_data(self, training_df=None, test_df=None):
    if not training_df is None:
      self.training_df, self.training_target = self.clean(training_df)
    if not test_df is None:
      self.test_df, self.test_target = self.clean(test_df)
  
  def clean(self, data):
    if data is None:
      return None, None
    df = data.astype({'text': 'U'})
    target = np.array(df['label'])
    df = pd.DataFrame(df['text'])
    return df, target
  
  def train_model(self, num_batch=16, num_epoch=25):
    training = tf.data.Dataset.from_tensor_slices((self.training_df['text'].values, self.training_target))
    test = tf.data.Dataset.from_tensor_slices((self.test_df['text'].values, self.test_target))
    
    hub_layer = hub.KerasLayer(self.embedding, input_shape=[], 
                              dtype=tf.string, trainable=True)

    self.model = tf.keras.Sequential()
    self.model.add(hub_layer)
    self.model.add(tf.keras.layers.Dense(16, activation='relu'))
    self.model.add(tf.keras.layers.Dense(8, activation='relu'))
    self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    self.model.summary()

    self.model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['binary_accuracy'])
    self.model.fit(training.shuffle(len(self.training_df)).batch(num_batch), epochs=num_epoch, verbose=1)
    self.cutoff = self.model.evaluate(test.batch(num_batch), verbose=0)[1]

  def test_model(self, num_batch=16):
    test = tf.data.Dataset.from_tensor_slices((self.test_df['text'].values, self.test_target))
    results = self.model.evaluate(test.batch(num_batch), verbose=1)
    for name, value in zip(self.model.metrics_names, results):
      print("%s: %.3f" % (name, value))
    
    prediction = self.model.predict(test.batch(num_batch))[:,0]
    prediction_df = self.test_df.copy()
    prediction_df['prediction'] = prediction > self.cutoff
    prediction_df['prediction'] = prediction_df['prediction'].map(lambda x: 1 if x else 0)
    prediction_df['actual'] = self.test_target

    print('Fraction of false positives: {}'.format(len(prediction_df[prediction_df['prediction'] > prediction_df['actual']])/len(prediction_df)))
    print('Fraction of false positives: {}'.format(len(prediction_df[prediction_df['prediction'] < prediction_df['actual']])/len(prediction_df)))

  def predict_model(self, input_data, num_batch=16):
    df = input_data.astype({'text': 'U'})
    df = pd.DataFrame(df['text'])
    df_data = tf.data.Dataset.from_tensor_slices(df['text'].values)
    prediction = self.model.predict(df_data.batch(num_batch))[:,0]
    prediction_df = df.copy()
    prediction_df['prediction'] = prediction > self.cutoff
    prediction_df['prediction'] = prediction_df['prediction'].map(lambda x: 1 if x else 0)
    return prediction_df
  
  def save_model(self, save_as='', directory=''):
    name = save_as if not save_as == '' else self.name
    self.model.save('saved_model\\' + name)

#client = MongoClient('mongodb+srv://haspburn71280:H8IsNoGood@hatebaddb-kbv0e.gcp.mongodb.net/test?retryWrites=true&w=majority')
#db = client.hatebad #database
#training_df = pd.DataFrame(db.hateTrainingSet.find())
#training_df['label'] = training_df['label'].map(lambda x: 1 if x == 'hate' else 0)
#test_df = pd.DataFrame(db.hateTestSet.find())
#test_df['label'] = test_df['label'].map(lambda x: 1 if x == 'hate' else 0)

#predict_df = pd.DataFrame(db.hateTestSet.find())
#predict_df['label'] = predict_df['label'].map(lambda x: 1 if x == 'hate' else 0)

#model = TweetModel(name='hate', training_df=training_df, test_df=test_df)
#model.train_model(num_epoch=5)
#model.test_model()
#model.predict_model(predict_df)
#model.save_model()

#model2 = TweetModel(load_file=True, name='hate')
#print(model2.predict_model(predict_df))
