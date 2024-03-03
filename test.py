import argparse
import random
import os
import tensorflow as tf
import numpy as np

from main import get_model
from data_iterator import DataIterator
from metrics_rs import evaluate_full

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='ml1m', help='book | taobao')
parser.add_argument('--gpu', type=str, default='0', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--item_count', type=int, default=1000)
parser.add_argument('--user_count', type=int, default=1000)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--category_num', type=int, default=2)
parser.add_argument('--topic_num', type=int, default=10)
parser.add_argument('--neg_num', type=int, default=10)
parser.add_argument('--cpt_feat', type=int, default=1)
parser.add_argument('--model_type', type=str, default='SINE', help='SINE|DNN|GRU4REC|MIND|ComiRec-DR|Model_ComiRec_SA')
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
    help="-1 for Baselines, "
        + "0 for Long-Intent, "
        + "1 for Self-supervised Learning, "
        + "2 for Long-Intent without gate unit and label attention, "
        + "3 for Long-Intent without gate unit, "
        + "4 for Long-Intent without label attention")
parser.add_argument('--temperature', type=float, default=1.0, help="softmax temperature (default:  1.0) - not studied.")
parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")

def test(test_file, log_path, best_model_path, similarity_model_path, args):
    dataset = args.dataset
    batch_size = args.batch_size
    maxlen = args.maxlen
    item_count = args.item_count
    user_count = args.user_count
    model_type = args.model_type
    topk = [10, 50, 100]
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, args)
    print('---> Start testing...')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, similarity_model_path, args.similarity_model_name, \
            args.dataset, batch_size, maxlen, train_flag=1)

        metrics = evaluate_full(sess, test_data, model, args)
        with open(log_path + '/evaluation_results_man.txt', 'w') as file:
            for k in range(len(topk)):
                result_str = '!!!! Test result topk=%d hitrate=%.4f ndcg=%.4f recall=%.4f \n' \
                    % (topk[k], metrics['hitrate'][k], metrics['ndcg'][k], metrics['recall'][k])
                print(result_str)
                file.write(result_str)

def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

def read_params_from_file(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            if value.isdigit():
                params[key] = int(value)
            elif '.' in value and all(c.isdigit() or c == '.' for c in value):
                params[key] = float(value)
            else:
                params[key] = value
    return params


if __name__ == '__main__':
    global_iter = 0
    args = parser.parse_args()
    params = read_params_from_file("/home/wangshengmin/workspace/SINE/log/SINE-li-2024-03-01 17:14/args.txt")
    args.__dict__.update(params)
    args.__dict__.update({"p": "test"})
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.dataset == 'taobao':
        path = './data/taobao/'
    if args.dataset == 'ml1m':
        path = './data/ml1m/'
    elif args.dataset == 'book':
        path = './data/book_data/'
    elif args.dataset == 'yzqytj':
        path = './data/yzqytj/'

    test_file = path + args.dataset + '_test.txt'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print_configuration(args)

    best_model_path = "/home/wangshengmin/workspace/SINE/log/SINE-li-2024-03-01 17:14/save_model/book_SINE_topic500_cept4_len20_neg10_unorm0_inorm0_catnorm0_head1_alpha0.5_beta0.0"
    # best_model_path = "/root/workspace/SINE_LOCAL/log/GRU4REC-baseline-2024-02-29 11:24/save_model/ml1m_GRU4REC_topic10_cept2_len20_neg10_unorm0_inorm0_catnorm0_head1_alpha0.0_beta0.0"
    log_path = os.path.dirname(os.path.dirname(best_model_path))
    similarity_model_path = "./data/ml1m/ml1m_ItemCF_IUF_similarity.pkl"
    test(test_file, log_path, best_model_path, similarity_model_path, args)
    print('--> Finish!')