import argparse
import random
import datetime
import time
import pytz
import os
import tensorflow as tf
import numpy as np

from data_iterator import DataIterator
from model_li import Model_SINE_LI
from model_ssl import Model_SINE_SSL
from model_li_ngl import Model_SINE_LI_NGL
from model_li_ng import Model_SINE_LI_NG
from model_li_nl import Model_SINE_LI_NL
from metrics_rs import evaluate_full

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='ml1m', help='book | taobao')
parser.add_argument('--gpu', type=str, default='0', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--item_count', type=int, default=1000)
parser.add_argument('--user_count', type=int, default=1000)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--category_num', type=int, default=2)
parser.add_argument('--topic_num', type=int, default=10)
parser.add_argument('--neg_num', type=int, default=10)
parser.add_argument('--cpt_feat', type=int, default=1)
parser.add_argument('--model_type', type=str, default='SINE', help='SINE')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--alpha', type=float, default=0.0, help='hyperparameter for interest loss (default: 0.0)')
parser.add_argument('--beta', type=float, default=0.0, help='hyperparameter for contrastive loss (default: 0.0)')
parser.add_argument('--batch_size', type=int, default=128, help='(k)')
parser.add_argument('--maxlen', type=int, default=20, help='(k)')
parser.add_argument('--epoch', type=int, default=30, help='(k)')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--coef', default=None)
parser.add_argument('--test_iter', type=int, default=50)
parser.add_argument('--user_norm', type=int, default=0)
parser.add_argument('--item_norm', type=int, default=0)
parser.add_argument('--cate_norm', type=int, default=0)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--output_size', type=int, default=128)
parser.add_argument('--experiment', type=int, default=0, \
    help="0 for Long-Intent, "
        + "1 for Self-supervised Learning, "
        + "2 for Long-Intent without gate unit and label attention, "
        + "3 for Long-Intent without gate unit, "
        + "4 for Long-Intent without label attention")
parser.add_argument('--temperature', type=float, default=1.0, help="softmax temperature (default:  1.0) - not studied.")
parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")

exp_dict = {
    0: ("li", Model_SINE_LI), 
    1: ("ssl", Model_SINE_SSL), 
    2: ("li-ngl", Model_SINE_LI_NGL),
    3: ("li-ng", Model_SINE_LI_NG),
    4: ("li-nl", Model_SINE_LI_NL)
}

def get_model(dataset, model_type, item_count, user_count, args):
    global exp_dict
    if not model_type == 'SINE':
        print("Invalid model_type : %s", model_type)
        return
    if args.experiment == 1:
        model = exp_dict[args.experiment][1](item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.maxlen, 
                               args.topic_num, args.category_num, args.alpha, args.beta, args.neg_num, args.cpt_feat, 
                               args.user_norm, args.item_norm, args.cate_norm, args.n_head)
    else:
        model = exp_dict[args.experiment][1](item_count, user_count, args.embedding_dim, args.hidden_size, args.output_size,
                        args.batch_size, args.maxlen, args.topic_num, args.category_num, args.alpha, args.neg_num,
                        args.cpt_feat, args.user_norm, args.item_norm, args.cate_norm, args.n_head)
    return model

def test(train_file, valid_file, test_file, log_path, best_model_path, similarity_model_path, args):
    dataset = args.dataset
    batch_size = args.batch_size
    maxlen = args.maxlen
    item_count = args.item_count
    user_count = args.user_count
    model_type = args.model_type
    topic_num = args.topic_num
    concept_num = args.category_num
    patience = args.patience
    test_iter = args.test_iter
    topk = [10, 50, 100]
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, args)
    print('---> Start testing...')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, similarity_model_path, args.similarity_model_name, \
            args.dataset, batch_size, maxlen, train_flag=1)

        metrics = evaluate_full(sess, test_data, model, args.embedding_dim)
        with open(log_path + '/evaluation_results.txt', 'w') as file:
            for k in range(len(topk)):
                result_str = '!!!! Test result topk=%d hitrate=%.4f ndcg=%.4f \n' \
                    % (topk[k], metrics['hitrate'][k], metrics['ndcg'][k])
                print(result_str)
                file.write(result_str)

def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


if __name__ == '__main__':
    global_iter = 0
    args = parser.parse_args()
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao/'
        args.item_count = 1708531
        args.user_count = 976780
        args.test_iter = 1000
    if args.dataset == 'ml1m':
        path = './data/ml1m/'
        args.item_count = 3706
        args.user_count = 6040
        args.test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        args.user_count = 603669
        args.item_count = 367983    
        args.test_iter = 1000
    elif args.dataset == 'yzqytj':
        path = './data/yzqytj/'
        args.user_count = 300890
        args.item_count = 286411
        args.test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print_configuration(args)

    best_model_path = "/home/wangshengmin/workspace/SINE/log/ssl-2024-02-24 15:23/save_model/ml1m_SINE_topic10_cept2_len20_neg10_unorm0_inorm0_catnorm0_head1_alpha0.0_beta1.0"
    log_path = os.path.dirname(os.path.dirname(best_model_path))
    similarity_model_path = "./data/ml1m/ml1m_ItemCF_IUF_similarity.pkl"
    test(train_file, valid_file, test_file, log_path, best_model_path, similarity_model_path, args)
    print('--> Finish!')