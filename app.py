import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.callbacks import EarlyStopping
from model import bert
from transformer import Transformer
from utils import common

__config = common.read_configs()
t = Transformer()
x_train, x_val, x_test, y_train, y_val, y_test = t.train_val_test_split()


def model_fit():

    multiclass_model = bert.build_bert_model()
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = tf.metrics.Accuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=__config['bert_config']['learning_rate'])

    multiclass_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    early_stopping = EarlyStopping(patience=3)

    history = multiclass_model.fit(x=tf.convert_to_tensor(x_train), y=tf.convert_to_tensor(y_train),
                                   validation_data=(x_val, y_val), epochs=5, callbacks=[early_stopping])

    multiclass_model.save(__config['bert_config']['model_storage'])

    return history


