import math
import os
import pickle
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from utils import read_file
import numpy as np


class PythonCodeDataset(IterableDataset):
    def __init__(self, args, files, vocab, shuffle=False):
        super(PythonCodeDataset).__init__()

        self.args = args
        self.files = files
        self.seq_size = args.seq_size
        self.vocab = vocab
        self.limit = args.limit
        self.shuffle = shuffle

    def get_data_from_files(self):

        if self.shuffle:
            random.shuffle(self.files)
        if self.limit is not None:
            self.files = self.files[:self.limit]
        for datafile in self.files:
            try:
                text = read_file(self.args, datafile)
                # print(datafile)
                if self.args.use_python_vocabulary:
                    int_text = [self.vocab.to_index(w) for w in text]
                else:
                    int_text = self.vocab.encode(text).ids

                for i in range(0, len(int_text) - self.seq_size - 1, self.args.data_step):

                    sequence = int_text[i: i + self.seq_size]

                    if 'xlnet' in self.args.model_name:
                        sequence.append(self.args.vocab_size+1)
                        labels = np.zeros(self.seq_size)
                        labels = np.append(labels, int_text[i + self.seq_size + 1])
                    else:
                        labels = int_text[i + 1: i + self.seq_size + 1]

                    yield torch.tensor((sequence, labels), dtype=torch.long)
            except Exception as e:
                #print(e)
                continue

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            data_files = self.files
        else:  # in a worker process
            # split workload
            num_files = len(self.files)
            per_worker = int(math.ceil((num_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, num_files)
            data_files = self.files[iter_start:iter_end]

        return self.get_data_from_files()
def mask_tokens(self, inputs: torch.Tensor):

        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def collate(examples, vocabulary):
    if ~hasattr(vocabulary, 'pad_id') or vocabulary.pad_id is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=vocabulary.pad_id)


def create_dataloader(args, files, vocabulary):
    dataset = PythonCodeDataset(args, files, vocabulary)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=lambda ex: collate(ex, vocabulary)
    )

    return dataloader


def get_processed_files_splited(train_path='./dataset_paths/flask_train_files',
                                valid_path='./dataset_paths/flask_valid_files',
                                test_path='./dataset_paths/flask_test_files', directory=False):
    if not directory:
        with open(train_path, 'rb') as f:
            train_files = [tf for tf in pickle.load(f) if '_processed' in tf]
        with open(valid_path, 'rb') as f:
            valid_files = [tf for tf in pickle.load(f) if '_processed' in tf]
        with open(test_path, 'rb') as f:
            test_files = [tf for tf in pickle.load(f) if '_processed' in tf]
    else:
        train_files = [os.path.join(train_path, p) for p in os.listdir(train_path)]
        valid_files = [os.path.join(train_path, p) for p in os.listdir(valid_path)]
        test_files = [os.path.join(train_path, p) for p in os.listdir(test_path)]
    return train_files, valid_files, test_files
