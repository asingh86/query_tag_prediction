import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.callbacks import EarlyStopping
from model import bert
from transformer import Transformer
from utils import common
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score


class Executor:

    def __init__(self):
        self.__config = common.read_configs()
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.target_label = Transformer().train_val_test_split()

    def model_fit(self):
        """This function calls the model object and perform the model fitting"""
        multiclass_model = bert.build_bert_model()
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = tf.metrics.CategoricalAccuracy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__config['bert_config']['learning_rate'])

        multiclass_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)

        early_stopping = EarlyStopping(patience=self.__config['bert_config']['callback_patience'])

        history = multiclass_model.fit(x=tf.convert_to_tensor(self.x_train), y=tf.convert_to_tensor(self.y_train),
                                       validation_data=(self.x_val, self.y_val),
                                       epochs=self.__config['bert_config']['epochs'],
                                       callbacks=[early_stopping],
                                       batch_size=self.__config['bert_config']['batch_size'])

        # save model
        multiclass_model.save(self.__config['bert_config']['model_storage'])

        # save history
        path = os.path.join(self.__config['bert_config']['model_storage'], '/fitting_history')
        with open(f'{path}.pickle', 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def perform_inference(self):
        multiclass_model = tf.saved_model.load(self.__config['bert_config']['model_storage'])
        y_pred = multiclass_model(tf.convert_to_tensor(self.x_test))

        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(self.y_test, y_pred)

        return m.result().numpy()

