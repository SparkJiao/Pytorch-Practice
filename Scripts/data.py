import json
import nltk
import re
import torch
import os
from os import path

from torchtext import data
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import BucketIterator


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class CoQA:
    def __init__(self, args):

        self.w_ctx = 2

        self.TEXT = Field(batch_first=True, sequential=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = Field(sequential=False, use_vocab=False, unk_token=None, )
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.batch_field = {
            "story": ("story", self.TEXT),
            "story_char": ("story", self.CHAR),
            "question": ("question_word", self.TEXT),
            "question_char": ("question_char", self.CHAR),
            "span_text": ("span_text", self.TEXT),
            "span_start": ("span_start", self.LABEL),
            "span_end": ("span_end", self.LABEL),
        }
        # self.list_field = [("story", self.TEXT), ("story_char", self.CHAR), ("question", self.TEXT),
        #                    ("question_char", self.CHAR), ("span_text", self.TEXT), ("span_start", self.LABEL),
        #                    ("span_end", self.LABEL)
        #                    ]

        self.data_path = path.dirname(path.dirname(__file__)) + '/data/coqa/'
        self.train_path = path.dirname(path.dirname(__file__)) + '/data/coqa/coqa-train-v1.0.json'
        self.dev_path = path.dirname(path.dirname(__file__)) + '/data/coqa/coqa-dev-v1.0.json'
        self.train_example_path = path.dirname(path.dirname(__file__)) + '/data/coqa/train-example.json'
        self.dev_example_path = path.dirname(path.dirname(__file__)) + '/data/coqa/dev-example.json'

        self.read(self.w_ctx, self.train_path, 'train')
        self.read(self.w_ctx, self.dev_path, 'dev')

        print("build the dataset...")
        self.train_batches = TabularDataset(path=self.train_example_path, format='json', fields=self.batch_field)
        self.dev_batches = TabularDataset(path=self.dev_example_path, format='json', fields=self.batch_field)

        print("build the vocabulary...")
        self.CHAR.build_vocab(self.train_batches, self.dev_batches)
        self.TEXT.build_vocab(self.train_batches, self.dev_batches, vectors=GloVe(name='6B', dim=100))
        #     with open('out1.txt', 'w', encoding='utf-8') as f:
        #         f.write(str(dict(self.CHAR.vocab.stoi)))
        #     with open('out2.txt', 'w', encoding='utf-8') as f:
        #         f.write(str(dict(self.TEXT.vocab.stoi)))

        print("build iterators...")
        self.train_iter, self.dev_iter = BucketIterator.splits((self.train_batches, self.dev_batches),
                                                               batch_sizes=[args.train_batch_size, args.dev_batch_size],
                                                               device=args.gpu,
                                                               sort_key=lambda x: len(x.story_char))

    def read(self, n_w_ctx, path, flag):
        with open(path, 'r') as input_file:
            data_set = json.load(input_file)
            datas = data_set["data"]

        # TODO: may be pre-process for the words
        # TODO: confirm the entities in the dictionary

        print("reading dataset...")
        batches = list()
        for article in datas:
            story = article["story"]
            questions = article["questions"]
            answers = article["answers"]
            for index, question in enumerate(questions):
                # TODO: erase the characters which can't be encoded
                # TODO: calculate the span_end
                batches.append(dict(
                    [("story", story), ("story_char", story), ("question", question["input_text"]),
                     ("question_char", question["input_text"]), ("span_text", answers[index]["span_text"]),
                     ("span_start", answers[index]["span_start"]),
                     ("span_end", answers[index]["span_end"])]))
            if "additional_answers" in article:
                for index, question in enumerate(questions):
                    additional_answers = article["additional_answers"]
                    for key in additional_answers:
                        batches.append(dict([("story", story), ("story_char", story),
                                             ("question", question["input_text"]),
                                             ("question_char", question["input_text"]),
                                             ("span_text", additional_answers[key][index]["span_text"]),
                                             ("span_start", additional_answers[key][index]["span_start"]),
                                             ("span_end", additional_answers[key][index]["span_end"])]))

            # TODO: after pre-process add the history information into the batch using n_w_ctx

        print("rewrite the data to json file...")
        with open(self.data_path + flag + "-example.json", 'w', encoding='utf-8') as f:
            for batch in batches:
                json.dump(batch, f)
                print("", file=f)

    # return batches

    # def get_dataset(self, n_w_ctx):
    #     # batches = Dataset(examples=batches_examples, fields=self.batch_field)
    #     print("build the dataset...")
    #     batches = TabularDataset(path=self.dev_example_path, format='json', fields=self.batch_field)
    #     return batches

    # def get_vocab(self, batches: Dataset):
    #     print(len(batches.examples))
    #     print("build the vocabulary...")
    #     self.CHAR.build_vocab(batches)
    #     self.TEXT.build_vocab(batches, vectors=GloVe(name='6B', dim=100))
    #     with open('out1.txt', 'w', encoding='utf-8') as f:
    #         f.write(str(dict(self.CHAR.vocab.stoi)))
    #     with open('out2.txt', 'w', encoding='utf-8') as f:
    #         f.write(str(dict(self.TEXT.vocab.stoi)))


if __name__ == '__main__':
    coqa = CoQA()
    # examples = coqa.get_dataset(2, "test_json.json")
    # coqa.get_vocab(examples)
    # path = path.dirname(path.dirname(__file__))
    # print(path + '/data/coqa)
