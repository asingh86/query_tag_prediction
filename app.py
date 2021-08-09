import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.callbacks import EarlyStopping
from model import bert
from transformer import Transformer
from utils import common
import os
import pickle

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
                                   validation_data=(x_val, y_val), epochs=__config['bert_config']['epochs'],
                                   callbacks=[early_stopping], batch_size=__config['bert_config']['batch_size'])

    # save model
    multiclass_model.save(os.join(__config['bert_config']['model_storage'], '/model'))

    # save history
    #path = os.join(__config['bert_config']['model_storage'], '/fitting_history')
    #with open(f'{path}.pickle', 'wb') as handle:
    #    pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

    predictions = multiclass_model.predict(x_test, )

    return history, multiclass_model




