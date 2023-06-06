""" This module includes model operations """

import json
import os

import numpy as np
import torch
import yaml
from torch.nn.functional import pad
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from customer_satisfaction.customer_satisfaction.helper import get_absolute_folder_path


class CustomerSatisfaction:
    """
    Customer sentiment and intent finder
    """

    def __init__(self):

        tokenizer_path = get_absolute_folder_path("resources", ["bart_tokenizer"])
        model_path = get_absolute_folder_path("resources", ["bart_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        config_path = os.path.join(get_absolute_folder_path("config"), "config.yaml")
        with open(config_path, "r") as stream:
            config_y = yaml.safe_load(stream)

        self.sentiment_labels = config_y["sentiment_labels"]
        self.intent_labels = config_y["intent_labels"]
        self.padding_len = config_y["padding_length"]

        script_path = os.path.join(get_absolute_folder_path("data"), "sample_script.json")
        with open(script_path, 'r') as f:
            self.script_dict = json.load(f)

    def apply_hypothesis(self, text_list, label, class_type):
        """
        Apply textual entailment to BART model and return probability of entailment.

        Args:
            text_list (list of str): list of texts
            label (str): Label to calculate relation
            class_type (str): It explains what is target. "intent" or "sentiment"

        Returns:
            - probs (np.array) - Probability of labels being true
        """

        if class_type not in ["intent", "sentiment"]:
            raise ValueError("Given class type must be intent or sentiment")

        hypothesis = f"{class_type} is {label}"

        # Tokenize given texts and hypothesis
        token_list = []
        for text in text_list:
            x = self.tokenizer.encode(text, hypothesis, return_tensors='pt',
                                      truncation_strategy='only_first')
            # ID=1 is padding token
            x = pad(x, [0, self.padding_len - x.shape[1]], mode="constant", value=1)
            token_list.append(x)

        # To run faster, collect all tokens and run through model
        x = torch.concat(token_list)
        logits = self.model(x)[0]

        # Three results in here "contradiction", "neutral" and "entailment". Eliminate neutral and get softmax
        filtered_logits = logits[:, [0, 2]]
        probs = filtered_logits.softmax(dim=1)[:, 1].detach().numpy()

        return probs

    def get_script_sentiments(self):
        """
        Returns script texts and sentiments

        Returns:
            - sentiment_list (list of dict) - Texts and sentiments
        """

        text_list = [dialog_dict["text"] for dialog_dict in self.script_dict.values()]

        # Get entailment probabilities of sentiments: positive, negative, neutral
        sentiment_probabilities = np.array([self.apply_hypothesis(text_list=text_list,
                                                                  label=label,
                                                                  class_type="sentiment")
                                            for label in self.sentiment_labels])

        # Get label with maximum probability
        indices = np.argmax(sentiment_probabilities, axis=0)
        sentiment_dict = [{"text": text,
                           "sentiment": self.sentiment_labels[index]}
                          for index, text in zip(indices, text_list)]

        return sentiment_dict

    def get_script_intents(self):
        """
        Returns script texts and intents

        Returns:
            - intent_dict (dict) - Texts and intents
        """

        text_list = [dialog_dict["text"] for dialog_dict in self.script_dict.values()]

        # Get entailment probabilities of intents: capacity, screen, network_connection
        intent_probabilities = np.array([self.apply_hypothesis(text_list=text_list,
                                                               label=label,
                                                               class_type="intent")
                                         for label in self.intent_labels])

        # If intent probability increase threshold, get as intent
        indices = intent_probabilities > 0.7
        intent_dict = [{"text": text,
                        "intents": list(np.array(self.intent_labels)[index])}
                       for index, text in zip(indices.T, text_list)]

        return intent_dict
