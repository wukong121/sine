import os
import sys
import json
import random
from collections import defaultdict
import pandas as pd

random.seed(1230)

name = 'ml1m1'
filter_size = 5
if len(sys.argv) > 1:
    name = sys.argv[1]
if len(sys.argv) > 2:
    filter_size = int(sys.argv[2])

users = defaultdict(list)
item_count = defaultdict(int)
def read_from_amazon(source):
    with open(source, 'r') as f:
        for line in f:
            r = json.loads(line.strip())
            uid = r['reviewerID']
            iid = r['asin']
            item_count[iid] += 1
            ts = float(r['unixReviewTime'])
            users[uid].append((iid, ts))


def read_from_taobao(source):
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            uid = int(conts[0])
            iid = int(conts[1])
            if conts[3] != 'pv':
                continue
            item_count[iid] += 1
            ts = int(conts[4])
            users[uid].append((iid, ts))


def read_from_yzqytj(source):
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            uid = int(conts[1])
            iid = int(conts[2])
            
            timestamp = int(conts[6])
            users[uid].append((iid, timestamp))

def read_movielens_ratings(source):
    with open(source, 'r') as f:
        for line in f:
            # 以::为分隔符，Movielens数据集的rating.dat文件中的字段是用::分隔的
            conts = line.strip().split('::')
            # 解析数据
            uid = int(conts[0])
            iid = int(conts[1])
            timestamp = int(conts[3])
            item_count[iid] += 1

            # 将数据添加到用户字典中
            users[uid].append((iid, timestamp))


if name == 'book':
    read_from_amazon('reviews_Books_5.json')
elif name == 'taobao':
    read_from_taobao('UserBehavior.csv')
elif name == 'yzqytj':
    read_from_yzqytj('yzqytj_train_dataset.csv')
elif name == 'ml1m1':
    read_movielens_ratings('ratings.dat')
    

items = list(item_count.items())
# 按照item交易的频次进行降序排列
items.sort(key=lambda x:x[1], reverse=True)

# 筛选掉交易频次低于5次的物品
item_total = 0
for index, (iid, num) in enumerate(items):
    if num >= filter_size:
        item_total = index + 1
    else:
        break

item_map = dict(zip([items[i][0] for i in range(item_total)], list(range(0, item_total))))

user_ids = list(users.keys())
filter_user_ids = []
for user in user_ids:
    item_list = users[user]
    index = 0
    for item, timestamp in item_list:
        # 计算该用户的物品中有多少个在 item_map 中存在的物品
        if item in item_map:
            index += 1
    # 用户交互过的物品总数太少，也会被过滤掉
    if index >= filter_size:
        filter_user_ids.append(user)
user_ids = filter_user_ids

# random.shuffle(user_ids)
num_users = len(user_ids)
user_map = dict(zip(user_ids, list(range(num_users))))

def export_map(name, map_dict):
    with open(name, 'w') as f:
        for key, value in map_dict.items():
            f.write('%s,%d\n' % (key, value))


def export_data(name, user_list, max_time=2):
    total_data = 0
    with open(name, 'w') as f:
        for user in user_list:
            if user not in user_map:
                continue
            item_list = users[user]
            reserve_item_list = []
            for i, item_ in enumerate(item_list):
                if item_[0] in item_map:
                    reserve_item_list.append(item_)
            item_list = reserve_item_list
            item_list.sort(key=lambda x:x[1])
            # 为序列数据赋予新的索引
            item_list = [(item_list[i][0], i + 1) for i in range(len(item_list))]
            if max_time == 2:
                item_list = item_list[0:-2]
            elif max_time == 1:
                item_list = item_list[0:-1]
                if len(item_list) > 100:
                    item_list = item_list[-100:]
            else:
                item_list = item_list
                if len(item_list) > 100:
                    item_list = item_list[-100:]
            for item, timestamp in item_list:
                if item in item_map:
                    f.write('%d,%d,%d\n' % (user_map[user], item_map[item], timestamp))
                    total_data += 1
    return total_data


path = os.getcwd() + '/' + name
print('source path is ', path)
if not os.path.exists(path):
    os.mkdir(path)
print('Total user=%d items=%d' % (len(user_map), len(item_map)))

export_map(path + "/" + name + '_user_map.txt', user_map)
export_map(path + "/" + name + '_item_map.txt', item_map)

total_train = export_data(path + "/" + name + '_train.txt', user_ids, max_time=2)
total_valid = export_data(path + "/" + name + '_valid.txt', user_ids, max_time=1)
total_test = export_data(path + "/" + name + '_test.txt', user_ids, max_time=0)
print('total behaviors train=%d, valid=%d, test=%d: ' %(total_train, total_valid, total_test))
