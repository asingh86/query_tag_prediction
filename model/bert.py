import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.python.keras.models import Model
from utils import common

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


def persist_model(multiclass_model: Model) -> None:
    multiclass_model.save(__config['bert_config']['model_storage'])
