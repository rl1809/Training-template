"""Preprocess raw data and encode data for model"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

PRETRAINED_TOKENIZER_PATH = 'vinai/phobert-base'

def choose_tokenizer(tokenize_type):
    if tokenize_type == 'bert':
        return AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER_PATH)
    return Tokenizer(filters='?')

class EncodeData:
    """Transform text data to numberical to feed to model"""
    def __init__(self, tokenize_type=None):
        self.tokenizer = choose_tokenizer(tokenize_type)
        self.label_encoder = LabelEncoder()
        self.tokenize_type = tokenize_type
    
    def fit(self, data, label):
        if self.tokenize_type != 'bert':
            self.tokenizer.fit_on_texts(data)
        self.label_encoder.fit(label)

    def tokenize_text(self, text):
        if self.tokenize_type == 'bert':

            encoded_text = self.tokenizer(
                text,
                add_special_tokens = True,
                max_length = 256,
                return_token_type_ids = False,
                pad_to_max_length='right',
                return_attention_mask = True,
                return_tensors = 'pt'
            )
        else:
            encoded_text = self.tokenizer.texts_to_sequences(text)
        return encoded_text
    
    def encode_label(self, label):
        encoded_label = self.label_encoder.transform(label)
        return encoded_label

    def __call__(self, data, label):
        encoded_data = self.tokenize_text(data)
        encoded_label = self.encode_label([label])[0]
        return encoded_data, encoded_label



