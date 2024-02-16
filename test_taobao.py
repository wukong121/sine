import random
import argparse
import os
from SINE.data_iterator import DataIterator
from SINE.model_li import *
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
parser.add_argument('--alpha', type=float, default=0.0, help='')
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

def get_model(dataset, model_type, item_count, user_count, args):
    if model_type == 'SINE':
        model = Model_SINE_LI(item_count, user_count, args.embedding_dim, args.hidden_size, args.output_size,
                           args.batch_size, args.maxlen, args.topic_num, args.category_num, args.alpha, args.neg_num,
                            args.cpt_feat, args.user_norm, args.item_norm, args.cate_norm, args.n_head)
    else:
        print("Invalid model_type : %s", model_type)
        return
    return model

def test(train_file, valid_file, test_file, args):
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
    # exp_name = get_exp_name(dataset, model_type, topic_num, concept_num, maxlen)
    topk = [10, 50, 100]
    tf.reset_default_graph()

    best_model_path = "save_model/" + '%s' % dataset + '_%s' % model_type + '_topic%d' % topic_num \
                      + '_cept%d' % concept_num + '_len%d' % maxlen + '_neg%d' % args.neg_num \
                      + '_unorm%d' % args.user_norm + '_inorm%d' % args.item_norm + '_catnorm%d' % args.cate_norm\
                      + '_head%d' % args.n_head + '_alpha{}'.format(args.alpha)
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, args)
    print('---> Start testing...')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
        
        test_data = DataIterator(test_file, batch_size, maxlen, train_flag=1)

        metrics = evaluate_full(sess, test_data, model, args.embedding_dim)
        for k in range(len(topk)):
            print('!!!! Test result topk=%d hitrate=%.4f ndcg=%.4f' % (topk[k], metrics['hitrate'][k],
                                                                           metrics['ndcg'][k]))
            
if __name__ == '__main__':
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
        args.test_iter = 1000
    if args.dataset == 'ml1m':
        path = './data/ml1m/'
        args.category_num = 2
        args.topic_num = 10
        args.item_count = 3706
        args.user_count = 6040
        args.test_iter = 500
    elif args.dataset == 'book':
        path = './data/book/'
        args.item_count = 367983    
        args.test_iter = 1000
    elif args.dataset == 'yzqytj':
        path = './data/yzqytj/'
        args.item_count = 286411
        args.test_iter = 1000
    
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # print_configuration(args)

    # best_epoch = train(train_file, valid_file, test_file, args)

    test(train_file, valid_file, test_file, args)
    print('--> Finish!')