import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.keras.models import Model
from utils import common
import pickle
import matplotlib.pyplot as plt

__config = common.read_configs()


def build_bert_model() -> Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(__config['bert_config']['tfhub_handle_preprocess'], name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(__config['bert_config']['tfhub_handle_encoder'], trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(__config['bert_config']['dropout'])(net)
    net = tf.keras.layers.Dense(__config['bert_config']['final_layer_length'], activation=None, name='classifier')(net)

    return tf.keras.Model(text_input, net)


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
    plt.ylable('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylable('Loss')
    plt.legend()


