import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class CoQA:
    def __init__(self):
        pass

    def read(self, n_w_ctx, path):
        with open(path, 'r') as input_file:
            data_set = json.load(input_file)
            datas = data_set["data"]

        # TODO: may be pre-process for the words
        # TODO: confirm the entities in the dictionary

        batches = list()

        for article in datas:
            story = article["story"]
            questions = article["questions"]
            answers = article["answers"]
            for index, question in enumerate(questions):
                batch = dict()
                batch["story"] = story
                batch["question"] = question["input_text"]
                batch["input_text"] = answers[index]["input_text"]
                # TODO: erase the characters which can't be encoded
                batch["span_text"] = answers[index]["span_text"]
                batch["span_start"] = answers[index]["span_start"]
                # TODO: calculate the span_end
                # batch["span_end"] = answers[index]["span_end"]
                if "additional_answers" in article:
                    additional_answers = article["additional_answers"]
                    additional_answer = list()
                    for key in additional_answers:
                        additional_answer.append(additional_answers[key][index])
                    batch["additional_answer"] = additional_answer

                # TODO: after pre-process add the history information into the batch using n_w_ctx

                batches.append(batch)

            return batches
