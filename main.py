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
from model_ssl_copy import Model_SINE_SSL
from model_li_ngl import Model_SINE_LI_NGL
from model_li_ng import Model_SINE_LI_NG
from model_li_nl import Model_SINE_LI_NL
from model_sine import Model_SINE
from model_comirec import Model_DNN, Model_GRU4REC, Model_MIND, Model_ComiRec_DR, Model_ComiRec_SA
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

exp_dict = {
    -1: ("baseline", None), 
    0: ("li", Model_SINE_LI), 
    1: ("ssl", Model_SINE_SSL), 
    2: ("li-ngl", Model_SINE_LI_NGL),
    3: ("li-ng", Model_SINE_LI_NG),
    4: ("li-nl", Model_SINE_LI_NL),
    5: ("sine", Model_SINE)
}

def get_model(dataset, model_type, item_count, user_count, args):
    global exp_dict
    if model_type == 'SINE':
        if args.experiment == 1:
            model = exp_dict[args.experiment][1](item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.maxlen, 
                                args.topic_num, args.category_num, args.alpha, args.beta, args.neg_num, args.cpt_feat, 
                                args.user_norm, args.item_norm, args.cate_norm, args.n_head)
        elif args.experiment == 5:
            model = exp_dict[args.experiment][1](item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.maxlen, 
                                args.topic_num, args.category_num, args.alpha, args.neg_num, args.cpt_feat, 
                                args.user_norm, args.item_norm, args.cate_norm, args.n_head)
        else:
            model = exp_dict[args.experiment][1](item_count, user_count, args.embedding_dim, args.hidden_size, args.output_size,
                            args.batch_size, args.maxlen, args.topic_num, args.category_num, args.alpha, args.neg_num,
                            args.cpt_feat, args.user_norm, args.item_norm, args.cate_norm, args.n_head)
    elif model_type == 'DNN': 
        model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.maxlen)
    elif model_type == 'GRU4REC': 
        model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.maxlen)
    elif model_type == 'MIND':
        relu_layer = True if dataset == 'book' else False
        model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.category_num, args.maxlen, relu_layer=relu_layer)
    elif model_type == 'ComiRec-DR':
        model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.category_num, args.maxlen)
    elif model_type == 'ComiRec-SA':
        model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, args.batch_size, args.category_num, args.maxlen)
    else:
        print ("Invalid model_type: ", model_type)
        return
    return model

def get_exp_name(dataset, model_type, topic_num, concept_num, maxlen, save=True):
    para_name = '_'.join([dataset, model_type, 't'+str(topic_num), 'c'+str(concept_num), 'len'+str(maxlen)])
    return para_name

def train(train_file, valid_file, test_file, log_path, best_model_path, similarity_model_path, args):
    global global_iter
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
    best_metric = 0
    best_metric_ndcg = 0
    best_epoch = 0
    summary_writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(
            train_file, similarity_model_path, args.similarity_model_name, args.dataset, batch_size, maxlen, train_flag=0)
        valid_data = DataIterator(
            valid_file, similarity_model_path, args.similarity_model_name, args.dataset, batch_size, maxlen, train_flag=1)
        test_data = DataIterator(
            test_file, similarity_model_path, args.similarity_model_name, args.dataset, batch_size, maxlen, train_flag=1)
        
        model = get_model(dataset, model_type, item_count, user_count, args)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('---> Start training...')

        for epoch in range(args.epoch):
            print('--> Epoch {} / {}'.format(epoch, args.epoch))
            trials = 0
            iter = 0
            loss_iter = 0.0
            start_time = time.time()
            while True:
                try:
                    hist_item, nbr_mask, i_ids, user_id, hist_item_list_augment, _ = train_data.next()  # (128,20), (128,20), (128,), (128,), (2,128,20), (128,)
                except StopIteration:
                    metrics = evaluate_full(sess, test_data, model, args)
                    for k in range(len(topk)):
                        print('!!!! Test result epoch %d topk=%d hitrate=%.4f ndcg=%.4f' \
                            % (epoch, topk[k], metrics['hitrate'][k], metrics['ndcg'][k]))
                    break
                if args.experiment == 1:
                    loss, summary = model.train(sess, hist_item, nbr_mask, i_ids, hist_item_list_augment)
                elif args.experiment == -1:
                    loss, summary = model.train(sess, [user_id, i_ids, hist_item, nbr_mask, args.learning_rate])
                elif args.experiment == 5:
                    loss, summary = model.train(sess, hist_item, nbr_mask, i_ids)
                else:
                    loss, summary = model.train(sess, hist_item, nbr_mask, i_ids, user_id)
                loss_iter += loss
                iter += 1
                global_iter += 1
                if iter % test_iter == 0:
                    print('--> Epoch {} / {} at iter {} loss {}'.format(epoch, args.epoch, iter, loss))
                summary_writer.add_summary(summary, global_iter)
                
            metrics = evaluate_full(sess, valid_data, model, args)
            for k in range(len(topk)):
                print('!!!! Validate result topk=%d hitrate=%.4f ndcg=%.4f' % (topk[k], metrics['hitrate'][k],
                                                                               metrics['ndcg'][k]))
            if 'hitrate' in metrics:
                hitrate = metrics['hitrate'][0]
                # ndcg = metrics['ndcg'][0]
                hitrate2 = metrics['ndcg'][1]
                if hitrate >= best_metric and hitrate2 >= best_metric_ndcg:
                    best_metric = hitrate
                    best_metric_ndcg = hitrate2
                    # best_metric_ndcg = ndcg
                    model.save(sess, best_model_path)
                    trials = 0
                    best_epoch = epoch
                    print('---> Current best valid hitrate=%.4f ndcg=%.4f' % (best_metric, best_metric_ndcg))
                else:
                    trials += 1
                    if trials > patience:
                        break

            test_time = time.time()
            print("time interval for one epoch: %.4f min" % ((test_time - start_time) / 60.0))
        summary_writer.close()    
    print('!!! Best epoch is %d' % best_epoch)
    return best_epoch

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
    # exp_name = get_exp_name(dataset, model_type, topic_num, concept_num, maxlen)
    topk = [10, 50, 100]
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, item_count, user_count, args)
    print('---> Start testing...')

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, similarity_model_path, \
            args.similarity_model_name, args.dataset, batch_size, maxlen, train_flag=1)

        metrics = evaluate_full(sess, test_data, model, args)
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

def save_configuration(args, file_path):
    with open(file_path, 'w') as file:
        for key, value in vars(args).items():
            file.write("{}: {}\n".format(key, value))


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

    utc_now = datetime.datetime.utcnow()
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    shanghai_time = utc_now.replace(tzinfo=pytz.utc).astimezone(shanghai_tz).strftime("%Y-%m-%d %H:%M")

    log_path = "./log/{}-{}-".format(args.model_type, exp_dict[args.experiment][0])+shanghai_time
    best_model_path = log_path + "/save_model/" + '%s' % args.dataset + '_%s' % args.model_type
    similarity_model_path = os.path.join(path,\
        args.dataset+"_"+args.similarity_model_name+"_similarity.pkl")
    print("similarity_model_path = ", similarity_model_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print_configuration(args)
    os.makedirs(log_path)
    save_configuration(args, log_path+"/args.txt")

    best_epoch = train(train_file, valid_file, test_file, log_path, best_model_path, similarity_model_path, args)
    test(train_file, valid_file, test_file, log_path, best_model_path, similarity_model_path, args)
    print('--> Finish!')