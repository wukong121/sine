import random
import numpy as np
from data_augmentation import *
from similarity_model import OfflineItemSimilarity
from typing import *


class DataIterator:

    def __init__(self, source, similarity_model_path, similarity_model_name, dataset_name,
                 batch_size=128,
                 maxlen=20,
                 train_flag=0,
                 shuffle=True
                ):
        self.source = source
        self.similarity_model_path = similarity_model_path
        self.similarity_model_name = similarity_model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0
        self.edge_count = 0
        self.edges = []
        self.shuffle = shuffle
        self.read(source)
        self.users = list(self.users)
        self.items = list(self.items)
        self.n_views = 3
        # self.similarity_model = OfflineItemSimilarity(
        #     data_file=self.source, 
        #     similarity_path=self.similarity_model_path, 
        #     model_name=self.similarity_model_name, 
        #     dataset_name=self.dataset_name
        # )
        # self.data_augmentation_methods = [Mask(), Reorder(), Insert(self.similarity_model), Substitute(self.similarity_model), Crop()]
        self.data_augmentation_methods = [Mask(), Reorder(), Crop()]
        self.augmentation_idx_list = list(itertools.combinations([i for i in range(self.n_views)], 2))
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0
        if train_flag == 0:
            self._shuffle()

    def _shuffle(self):
        random.shuffle(self.edges)

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph: Dict[int, List[Tuple[int, int, int]]] = {}
        self.users = set()
        self.items = set()
        line_cnt = 0
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((user_id, item_id, time_stamp))
                line_cnt += 1
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[2])  # 按时间戳排序
            self.edges.extend(value[3:])
            self.graph[user_id] = [(usr_, itm_, time_) for usr_, itm_, time_ in value]  # 把排序和截断好的数据重新赋值给user_id
        self.edge_count = len(self.edges)
    
    def __next__(self):
        item_time_list = []
        if self.train_flag == 0:  # 训练模式
            if self.index + self.eval_batch_size > self.edge_count:
                self.index = 0
                self._shuffle()
                raise StopIteration
            edges_list = self.edges[self.index: self.index+self.eval_batch_size]
            self.index += self.eval_batch_size
            user_id_list, item_id_list, item_time_list = zip(*edges_list)
        else:  # 测试模式
            total_user = len(self.users)
            if self.index > total_user:
                self.index = 0
                raise StopIteration
            end_ = min(self.index + self.eval_batch_size, total_user)
            user_id_list = self.users[self.index: end_]
            item_id_list = [self.graph[user_][-1][1] for user_ in user_id_list]  # 获取每个用户最后交互过的那个物品
            self.index += self.eval_batch_size

        # item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for i, user_id in enumerate(user_id_list):
            item_list = self.graph[user_id]   # item_list其实是u,i,t的元组
            item_ = [_item[1] for _item in item_list]
            # k是指历史序列的截取位置
            if self.train_flag == 0:  # 训练模式
                k = item_list.index((user_id_list[i], item_id_list[i], item_time_list[i]))
            else:  # 测试模式
                k = item_list.index(item_list[-1])
            if k >= self.maxlen:
                hist_item_list.append(item_[k-self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                # hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_item_list.append(item_[:k] + [0] * (self.maxlen - k)) # 前面k个item保留，后面不足maxlen的位置补0
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))  # 前面k个item的mask为1，后面不足maxlen的位置为0.0
        
        hist_item_list_aug1, hist_item_list_aug2 = [], []
        for i, seq in enumerate(hist_item_list):
            augment_idx = i % self.total_augmentation_samples
            hist_item_list_aug1.append(self.align(self.data_augmentation_methods[self.augmentation_idx_list[augment_idx][0]](seq)))
            hist_item_list_aug2.append(self.align(self.data_augmentation_methods[self.augmentation_idx_list[augment_idx][1]](seq)))

        hist_item_list_augment = []
        hist_item_list_augment.append(hist_item_list_aug1)
        hist_item_list_augment.append(hist_item_list_aug2)

        return np.array(hist_item_list), np.array(hist_mask_list), np.array(item_id_list), np.array(user_id_list), np.array(hist_item_list_augment)

    def align(self, seq):
        pad_len = self.maxlen - len(seq)
        seq = [0] * pad_len + seq
        seq = seq[-self.maxlen:]
        assert len(seq) == self.maxlen
        return seq


class LargeDataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=20,
                 train_flag=0,
                 shuffle=True
                 ):
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0
        self.index_valid = 0
        self.index_test = 0
        self.edge_count = 0
        self.edges = []
        self.shuffle = shuffle
        self.read(source)
        self.users = list(self.users)
        self.items = list(self.items)
        print('edges: %d users: %d items: %d' % (self.edge_count, len(self.users), len(self.items)))
        self._shuffle()

    def _shuffle(self):
        random.shuffle(self.edges)

    def __iter__(self):
        return self

    def next(self, flag=0):
        return self.__next__(flag)

    def read(self, source):
        self.graph = {}
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((user_id, item_id, time_stamp))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[2])
            self.edges.extend(value[3:-2])
            self.graph[user_id] = [(usr_, itm_, time_) for usr_, itm_, time_ in value]
        self.edge_count = len(self.edges)

    def __next__(self, flag):
        item_time_list = []
        if flag == 0:
            if self.index + self.eval_batch_size > self.edge_count:
                self.index = 0
                self._shuffle()
                raise StopIteration
            edges_list = self.edges[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size
            user_id_list, item_id_list, item_time_list = zip(*edges_list)
        elif flag == 1:
            total_user = len(self.users)
            if self.index_valid > total_user:
                self.index_valid = 0
                raise StopIteration
            end_ = min(self.index_valid + self.eval_batch_size, total_user)
            user_id_list = self.users[self.index_valid: end_]
            item_id_list = [self.graph[user_][-2][1] for user_ in user_id_list]
            self.index_valid += self.eval_batch_size
        else:
            total_user = len(self.users)
            if self.index_test > total_user:
                self.index_test = 0
                raise StopIteration
            end_ = min(self.index_test + self.eval_batch_size, total_user)
            user_id_list = self.users[self.index_test: end_]
            item_id_list = [self.graph[user_][-1][1] for user_ in user_id_list]
            self.index_test += self.eval_batch_size

        # item_id_list = []
        hist_item_list = []
        hist_mask_list = []
        for i, user_id in enumerate(user_id_list):
            item_list = self.graph[user_id]
            item_ = [_item[1] for _item in item_list]
            if flag == 0:
                k = item_list.index((user_id_list[i], item_id_list[i], item_time_list[i]))
            elif flag == 1:
                k = item_list.index(item_list[-2])
            else:
                k = item_list.index(item_list[-1])
            if k >= self.maxlen:
                hist_item_list.append(item_[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
            else:
                # hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_item_list.append(item_[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))
        return np.array(hist_item_list), np.array(hist_mask_list), np.array(item_id_list)
