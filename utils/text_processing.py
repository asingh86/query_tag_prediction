import string
import demoji
import nltk
from utils import common

# download necessary package
nltk.download('punkt')


class TextProcessing:

    def __init__(self):
        self.__config = common.read_configs()

    @staticmethod
    def convert_lowercase(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_emoji(text: str) -> str:
        return demoji.replace(text, "")

    @staticmethod
    def remove_punctuation(text: str) -> str:
        for p in string.punctuation:
            text = text.replace(p, "")
        return text

    def remove_manual_characters(self, text: str) -> str:
        for s in self.__config['filters']['manual_characters_list']:
            text = text.replace(s, '')
        return text

    def remove_manual_stopwords(self, text: str) -> str:
        token_sentence = nltk.word_tokenize(text)
        filtered_text = [_word for _word in token_sentence if
                         _word not in self.__config['filters']['manual_stopwords_list']]
        return ' '.join(filtered_text)

    def process_text(self, text: str) -> list:
        """Process the text by removing emojis, punctuation etc"""

        processed_text = str(text)
        if self.__config['filters']['lowercase']:
            processed_text = self.convert_lowercase(processed_text)
        if self.__config['filters']['manual_characters']:
            processed_text = self.remove_manual_characters(processed_text)
        if self.__config['filters']['emoji']:
            processed_text = self.remove_emoji(processed_text)
        if self.__config['filters']['punctuation']:
            processed_text = self.remove_punctuation(processed_text)
        if self.__config['filters']['manual_stopwords']:
            processed_text = self.remove_manual_stopwords(processed_text)

        return processed_text
