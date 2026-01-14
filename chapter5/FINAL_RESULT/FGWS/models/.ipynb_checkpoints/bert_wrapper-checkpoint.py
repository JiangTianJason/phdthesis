"""
Parts based on https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
"""
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils import print_model_state_dict, list_join


class BertWrapper:
    def __init__(self, config, logger, tokenizer, model):
        self.config = config
        self.logger = logger

        self.tokenizer = tokenizer

        self.model = model

        print_model_state_dict(logger, self.model)

    def pre_pro(self, text):
        assert isinstance(text, list)

        tokens = self.tokenizer.encode_plus(
            list_join(text),
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
        )

        return tokens["input_ids"], tokens["attention_mask"]
