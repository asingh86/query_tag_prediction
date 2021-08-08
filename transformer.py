import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.text_processing import TextProcessing
from utils import common


class Transformer:

    def __init__(self, seed: int = 123, buffer_size: int = 7000, batch_size: int = 32,
                 train_sample: float = 0.8, validation_sample=0.2, test_sample=0.1):
        self.__config = common.read_configs()
        self.__seed = seed
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__train_sample = train_sample
        self.__validation_sample = validation_sample
        self.__test_sample = test_sample

    def read_and_clean_data(self):
        queries = []
        t = TextProcessing()
        with open(self.__config['file_path'], ) as f:
            for line in f:
                load_line = json.loads(line)
                load_line['processed_message'] = t.process_text(load_line['original_message'])
                queries.append(load_line)

        return queries

    def one_hot_target_transformation(self, target: str = 'tag1'):
        queries = self.read_and_clean_data()
        df = pd.DataFrame(queries)
        one_hot = pd.get_dummies(df[target])
        df = df.join(one_hot)
        return df

    def get_feature(self):
        df = self.one_hot_target_transformation()

        text = []
        target = []
        df_np = df.to_numpy()
        target_label = df.columns[5:]
        for i in range(len(df)):
            text.append(df_np[i][4])
            target.append(df_np[i][5:])
        return text, target, target_label

    def train_val_test_split(self):

        text, target, target_label = self.get_feature()

        train_sample = len(text) * self.__config['parameters']['train_sample']
        val_sample = len(text) * self.__config['parameters']['validation_sample']
        val_sample = val_sample / train_sample

        x_train, x_test, y_train, y_test = train_test_split(text, target,
                                                            test_size=self.__config['parameters']['test_sample'],
                                                            random_state=self.__config['parameters']['seed'])
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_sample,
                                                          random_state=self.__config['parameters']['seed'])

        x_train = np.array(x_train)
        x_val = np.array(x_val)
        x_test = np.array(x_test)
        y_train = np.array(y_train).astype(np.float32)
        y_val = np.array(y_val).astype(np.float32)
        y_test = np.array(y_test).astype(np.float32)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def get_tensorflow_dataset(self):

        x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split()

        x_train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_train))
        x_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_val))
        x_test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_test))
        y_train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_train))
        y_val = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_val))
        y_test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_test))

        return x_train, x_val, x_test, y_train, y_val, y_test
