"""Preprocess raw data and encode data for model"""

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer


class EncodeData:

    def __init__(self, tokenizer, label_encoder, tokenize_type=None):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.tokenize_text = tokenize_type



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
        encode_data = self.en



