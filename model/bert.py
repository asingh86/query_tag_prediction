import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from utils import common
import pickle
import os
import matplotlib.pyplot as plt
from typing import Any, Dict
import numpy as np


class BertModel:

    def __init__(self):
        self.__config = common.read_configs()

    def build_bert_model(self) -> Model:
        """build the bert model from tensorflow hub, the model also handles the text preprocessing as well"""
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.__config['bert_config']['tfhub_handle_preprocess'], name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.__config['bert_config']['tfhub_handle_encoder'], trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(self.__config['bert_config']['dropout'])(net)
        net = tf.keras.layers.Dense(self.__config['bert_config']['final_layer_length'], activation='softmax', name='classifier')(net)

        return tf.keras.Model(text_input, net)

    def model_fit(self, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> (Model, Dict):
        """This function calls the model object and perform the model fitting"""
        multiclass_model = self.build_bert_model()
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = tf.metrics.CategoricalAccuracy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__config['bert_config']['learning_rate'])

        multiclass_model.compile(optimizer=optimizer,
                                 loss=loss,
                                 metrics=metrics)

        early_stopping = EarlyStopping(patience=self.__config['bert_config']['callback_patience'])

        history = multiclass_model.fit(x=tf.convert_to_tensor(x_train), y=tf.convert_to_tensor(y_train),
                                       validation_data=(x_val, y_val),
                                       epochs=self.__config['bert_config']['epochs'],
                                       callbacks=[early_stopping],
                                       batch_size=self.__config['bert_config']['batch_size'])

        # save model
        multiclass_model.save(self.__config['bert_config']['model_storage'])

        # save history
        path = os.path.join(self.__config['bert_config']['model_storage'], '/fitting_history')
        with open(f'{path}.pickle', 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return multiclass_model, history

    def perform_inference(self, x_test: np.ndarray) -> float:
        multiclass_model = tf.saved_model.load(self.__config['bert_config']['model_storage'])
        y_pred = multiclass_model(tf.convert_to_tensor(self.x_test))

        return y_pred

    @staticmethod
    def visualise_model_fitting_history(history_dict: Dict[str, Any]) -> None:
        train_loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        train_accuracy = history_dict['accuracy']
        val_accuracy = history_dict['val_accuracy']

        epochs = range(1, len(train_accuracy) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_accuracy, 'r', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        return plt.show


