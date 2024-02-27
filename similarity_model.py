import os
import math
import pickle
from tqdm import tqdm


class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', \
        dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        self.itemSimBest = dict()
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score
    
    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        userid_pre = -1
        items = []
        for line in open(data_file).readlines():
            user_id, item, timestamp = line.strip().split(',')
            if userid_pre == -1:
                userid_pre = user_id
            if user_id != userid_pre:
                train_data_list.append(items)
                train_data_set_list += items
                for itemid in items:
                    train_data.append((userid_pre,itemid,int(1)))
                userid_pre = user_id
                items = []
            items.append(item)
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, seg_id=0, train=None):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        seg_len = int(len(train) / 4)  # split the train dataset to 4 part, distributed processing 
        if seg_id == 3:
            train = dict(list(train.items())[seg_id*seg_len:])
        else:
            train = dict(list(train.items())[seg_id*seg_len: (seg_id+1)*seg_len])
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item,{})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item,0);
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                        vector_size=20, window=5, min_count=0, 
                                        epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item,{})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item,0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
        elif self.model_name == 'LightGCN':
            # train a item embedding from lightGCN model, and then convert to sim dict
            print("generating similarity model..")
            itemSimBest = light_gcn.generate_similarity_from_light_gcn(self.dataset_name)
            print("LightGCN based model saved to: ", save_path)  

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            for seg_id in range(4):
                self._generate_item_similarity(seg_id=seg_id)
            self._save_dict(self.itemSimBest, save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k = top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))


if __name__ == "__main__":
    similarity_model = OfflineItemSimilarity(
        data_file="./data/book_data/book_train.txt", 
        similarity_path="./data/book_data/book_ItemCF_IUF_similarity.pkl", 
        model_name="ItemCF_IUF", 
        dataset_name="book"
    )
