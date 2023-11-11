import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from data.nlp import Tokenizer, Vocab
from data.language_utils import word_to_indices, letter_to_vec


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data




class Sent140Dataset(Dataset):
    def __init__(self,
                 client_id: int,
                 client_str: str,
                 data: list,
                 targets: list,
                 is_to_tokens: bool = True,
                 tokenizer: Tokenizer = None):
        """get `Dataset` for sent140 dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
            is_to_tokens (bool, optional), if tokenize data by using tokenizer
            tokenizer (Tokenizer, optional), tokenizer
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self.data_token = []
        self.data_tokens_tensor = []
        self.targets_tensor = []
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.fix_len = None

        self._process_data_target()
        if is_to_tokens:
            self._data2token()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = [e[4] for e in self.data]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def _data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def encode(self, vocab: 'Vocab', fix_len: int):
        """transform token data to indices sequence by `Vocab`
        Args:
            vocab (fedlab_benchmark.leaf.nlp_utils.util.vocab): vocab for data_token
            fix_len (int): max length of sentence
        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_tokens_tensor) > 0:
            self.data_tokens_tensor.clear()
            self.targets_tensor.clear()
        self.vocab = vocab
        self.fix_len = fix_len
        pad_idx = self.vocab.get_index('<pad>')
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_tokens_tensor.append(
                self.__encode_tokens(tokens, pad_idx))
        for target in self.targets:
            self.targets_tensor.append(torch.tensor(target))

    def __encode_tokens(self, tokens, pad_idx) -> torch.Tensor:
        """encode `fix_len` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than fix_len, it will use pad word for padding to fix_len
        if one sentence length is longer than fix_len, it will cut the first max_words words
        Args:
            tokens (list[str]): data after tokenizer
        Returns:
            integer list of indices with `fix_len` length for tokens input
        """
        x = [pad_idx for _ in range(self.fix_len)]
        for idx, word in enumerate(tokens[:self.fix_len]):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_tokens_tensor[item], self.targets_tensor[item]



class SSpeare(Dataset):
    def __init__(self, train=True):
        super(SSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/shakespeare/train",
                                                                                 "./data/shakespeare/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")




if __name__ == '__main__':
    test = SSpeare(train=True)
    x = test.get_client_dic()
    print(len(x))
    for i in range(100):
        print(len(x[i]))
