import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras

class Model:
  def __init__(self, data=None, test_data=None, target='score', name='something', load_data=False):
    self.name = name
    self.test_data = test_data
    self.data = data
    if load_data:
      self.model = None if data is None else self.create_model(data)
    else:
      self.model = tf.keras.models.load_model('saved_model\\' + self.name)

  def create_model(self, train_df=None, _target='score'):
    data = self.data if train_df is None else train_df

    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      data, data[_target], num_epochs=None, shuffle=True)
    predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      data, data[_target], shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
      key="text", 
      module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.keras.optimizers.Adagrad(lr=0.003),
      config=tf.estimator.RunConfig(model_dir='D:\\Hate Bad\\' + self.name))

    estimator.train(input_fn=train_input_fn, steps=5000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    estimator.export_saved_model('saved_model\\' + self.name)

    return estimator

  def test_model(self, test_df=None, _target='score'):
    data = self.test_data if test_df is None else test_df

    predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      data, data[_target], shuffle=False)
    
    test_eval_result = self.model.evaluate(input_fn=predict_test_input_fn)
    print(test_eval_result)
    print("Test set accuracy: {accuracy}".format(**test_eval_result))
  
  def predict(self, input_data, _target='score'):
    print(len(input_data))
    data_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      input_data, shuffle=False)
    result = self.model.predict(input_fn=data_input_fn)
    return [x for x in result]