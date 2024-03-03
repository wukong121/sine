import numpy as np
from deepctr.feature_column import concat_func
import tensorflow as tf
# from tensorflow.nn.rnn_cell import GRUCell

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class Model(object):
    def __init__(self, n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, share_emb=True, 
                 flag="DNN", item_norm=0):
        self.model_flag = flag
        self.reg = False
        self.user_eb = None
        self.batch_size = batch_size
        self.n_size = n_mid
        self.neg_num = 10
        self.lr = 0.001
        self.alpha_para = 0.0
        self.hist_max = seq_len
        self.dim = embedding_dim
        self.share_emb = share_emb
        self.item_norm = item_norm
        self.user_count = user_count
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / embedding_dim),
                                                               maxval=-np.sqrt(3 / embedding_dim))
        
        with tf.name_scope('Inputs'):
            self.i_ids = tf.placeholder(shape=[None], dtype=tf.int32)
            self.item = tf.placeholder(shape=[None, seq_len], dtype=tf.int32)
            self.nbr_mask = tf.placeholder(shape=[None, seq_len], dtype=tf.float32)
            self.user_id = tf.placeholder(shape=[None], name='user_id', dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.item_input_lookup = tf.get_variable("input_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.item_input_lookup_var = tf.get_variable("input_bias_lookup_table", [n_mid],
                                                       initializer=tf.zeros_initializer(), trainable=False)
            self.position_embedding = tf.get_variable(
                    shape=[1, self.hist_max, embedding_dim],
                    name='position_embedding')
            if self.share_emb:
                self.item_output_lookup = self.item_input_lookup
                self.item_output_lookup_var = self.item_input_lookup_var
            else:
                self.item_output_lookup = tf.get_variable("output_embedding_var", [n_mid, embedding_dim], trainable=True)
                self.item_output_lookup_var = tf.get_variable("output_bias_lookup_table", [n_mid],
                                                             initializer=tf.zeros_initializer(), trainable=False)
            
            # Trainable的参数
            self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=tf.zeros_initializer(),
                                                     shape=[user_count, embedding_dim], trainable=True)  # 这里的initializer这样是否可以，还是说应该按照源代码来
            self.the_first_w = tf.get_variable('the_first_w', initializer=self.initializer_param,
                                           shape=[embedding_dim, embedding_dim])
            self.the_first_bias = tf.get_variable('the_first_bias', initializer=self.initializer_param,
                                              shape=[embedding_dim])

        emb = tf.nn.embedding_lookup(self.item_input_lookup,
                                     tf.reshape(self.item, [-1]))
        self.item_emb = tf.reshape(emb, [-1, self.hist_max, self.dim])  # ?*20*128
        self.mask_length = tf.cast(tf.reduce_sum(self.nbr_mask, -1), dtype=tf.int32)

        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)  # ?*128
        self.item_output_emb = self.output_item2()

    def output_item2(self):
        if self.item_norm:
            item_emb = tf.nn.l2_normalize(self.item_output_lookup, dim=-1)
            return item_emb
        else:
            return self.item_output_lookup

    def _xent_loss(self, user):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        self.loss = tf.reduce_mean(loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        return loss

    def _xent_loss_weight(self, user, seq_multi):
        emb_dim = self.dim
        loss = tf.nn.sampled_softmax_loss(
            weights=self.output_item2(),
            # weights=self.item_output_lookup,
            biases=self.item_output_lookup_var,
            labels=tf.reshape(self.i_ids, [-1, 1]),
            inputs=tf.reshape(user, [-1, emb_dim]),
            num_sampled=self.neg_num * self.batch_size,
            num_classes=self.n_size,
            partition_strategy='mod',
            remove_accidental_hits=True
        )

        regs = self.calculate_interest_loss(seq_multi)

        self.loss = tf.reduce_mean(loss)
        self.reg_loss = self.alpha_para * tf.reduce_mean(regs)
        loss = self.loss + self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return loss
    
    def summary_loss(self):
        tf.summary.scalar('sampled_softmax_loss', self.loss)
        tf.summary.scalar('interest_loss', self.reg_loss)
        self.merged_summary = tf.summary.merge_all()

    def train(self, sess, hist_item, nbr_mask, i_ids, user_id):
        feed_dict = {
            self.i_ids: i_ids,
            self.item: hist_item,
            self.nbr_mask: nbr_mask,
            self.user_id: user_id
        }
        loss, _, summary = sess.run([self.loss, self.optimizer, self.merged_summary], feed_dict=feed_dict)
        return loss, summary

    def output_item(self, sess):
        item_embs = sess.run(self.item_output_emb)
        # item_embs = sess.run(self.item_output_lookup)
        return item_embs

    def output_user(self, sess, hist_item, nbr_mask, user_id):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.item: hist_item,
            self.nbr_mask: nbr_mask,
            self.user_id: user_id
        })
        return user_embs
    
    def save(self, sess, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + '_model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + '_model.ckpt')
        print('model restored from %s' % path)

    def calculate_interest_loss(self, user_interest):
        norm_interests = tf.nn.l2_normalize(user_interest, -1)
        dim0, dim1, dim2 = get_shape(user_interest)

        interests_losses = []
        for i in range(1, (dim1 + 1) // 2):
            roll_interests = tf.concat(
                    (norm_interests[:, i:, :], norm_interests[:, 0:i, :]), axis=1)
            # compute pair-wise interests similarity.
            interests_radial_diffs = tf.math.multiply(
                    tf.reshape(norm_interests, [dim0*dim1, dim2]),
                    tf.reshape(roll_interests, [dim0*dim1, dim2]))
            interests_loss = tf.math.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = tf.reshape(interests_loss, [dim0, dim1])
            interests_loss = tf.math.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        if dim1 % 2 == 0:
            half_dim1 = dim1 // 2
            interests_part1 = norm_interests[:, :half_dim1, :]
            interests_part2 = norm_interests[:, half_dim1:, :]
            interests_radial_diffs = tf.math.multiply(
                    tf.reshape(interests_part1, [dim0*half_dim1, dim2]),
                    tf.reshape(interests_part2, [dim0*half_dim1, dim2]))
            interests_loss = tf.math.reduce_sum(interests_radial_diffs, axis=-1)
            interests_loss = tf.reshape(interests_loss, [dim0, half_dim1])
            interests_loss = tf.math.reduce_sum(interests_loss, axis=-1)
            interests_losses.append(interests_loss)

        # NOTE(reed): the original interests_loss lay in [0, 2], so the
        # combination_size didn't divide 2 to normalize interests_loss into
        # [0, 1]
        self._interests_length = None
        if self._interests_length is not None:
            combination_size = tf.cast(
                    self._interests_length * (self._interests_length - 1),
                    tf.dtypes.DType(tf.float32))
        else:
            combination_size = dim1 * (dim1 - 1)
        interests_loss = 0.5 + (
                tf.math.reduce_sum(interests_losses, axis=0) / combination_size)

        return interests_loss

class CapsuleNetwork(tf.layers.Layer):
    def __init__(self, dim, seq_len, bilinear_type=2, num_interest=4, hard_readout=True, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope('bilinear'):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim, activation=None, bias_initializer=None)
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(item_his_emb, self.dim * self.num_interest, activation=None, bias_initializer=None)
            else:
                w = tf.get_variable(
                    'weights', shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer())
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, :self.seq_len, :, :] * u, axis=3)

        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim])
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim])

        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name='item_emb_hat_iter')
        else:
            item_emb_hat_iter = item_emb_hat

        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len]))
        else:
            capsule_weight = tf.stop_gradient(tf.truncated_normal([get_shape(item_his_emb)[0], self.num_interest, self.seq_len], stddev=1.0))

        for i in range(3):
            atten_mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_interest, 1])
            paddings = tf.zeros_like(atten_mask)

            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(tf.equal(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = tf.matmul(item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2]))
                delta_weight = tf.reshape(delta_weight, [-1, self.num_interest, self.seq_len])
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = tf.reshape(interest_capsule, [-1, self.num_interest, self.dim])

        if self.relu_layer:
            interest_capsule = tf.layers.dense(interest_capsule, self.dim, activation=tf.nn.relu, name='proj')

        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))

        if self.hard_readout:
            readout = tf.gather(tf.reshape(interest_capsule, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest)
        else:
            readout = tf.matmul(tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]), interest_capsule)
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])

        return interest_capsule, readout


class Model_SINE_LI(Model):
    def __init__(self, n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head, temperature):
        super(Model_SINE_LI, self).__init__(n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, 
                                         flag="SINE", item_norm=item_norm)
        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = temperature
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head
        self.output_units = output_size
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)  # ?,category_num,128
        self.user_eb_short = self.labeled_attention(self.seq_multi)  # ?*128

        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.item_emb,
                                                            self.the_first_w, self.the_first_bias)  # (?, 128)
        self.user_eb = self.gate_user_eb(self.long_user_embedding, self.user_eb_short, self.user_embedding)
        self._xent_loss_weight(self.user_eb, self.seq_multi)
        self.summary_loss()
    
    def gate_user_eb(self, long_user_embedding, user_eb_short, user_embedding):
        gate_input = concat_func([long_user_embedding, user_eb_short, user_embedding])  # (?, 128+128+128)
        gate = tf.keras.layers.Dense(self.output_units, activation='sigmoid')(gate_input)  # (?, 128)
        gate_output = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]) \
            + tf.multiply(1 - x[0], x[2]))([gate, user_eb_short, long_user_embedding])  # (?, 128)
        user_eb = self.l2_normalize(gate_output)  # (?, 128)
        return user_eb
        
    def l2_normalize(self, x, axis = -1):
        return tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)

    def attention_level_one(self, user_embedding, item_emb, the_first_w, the_first_bias):
        user_embedding_expanded = tf.expand_dims(user_embedding, axis=1)  # [n,1,128]，转置后变成[n,128,1]
        weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.sigmoid(tf.add(tf.matmul(item_emb, the_first_w), the_first_bias)),
            tf.transpose(user_embedding_expanded, perm=[0,2,1])), perm=[0,2,1]))  # [n,1,20]
        out = tf.reduce_sum(tf.multiply(item_emb, tf.transpose(weight, perm=[0,2,1])), axis=1)  # reduce_sum([n,20,128]multiply[n,20,1])---[n,128]
        return tf.reshape(out, [-1, self.dim])
    
    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(item_emb, self.nbr_mask)#[N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]，意图序列
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)  # D维，预测的下一个意图

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)，不同兴趣的聚合权重
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            return item_emb

    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, dim=1), axis=-1)#[-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = tf.reshape(items, [-1, self.dim])  # N,D
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])  # ?,2,128

        return seq


class Model_SINE_LI_NL(Model):
    def __init__(self, n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head, temperature):
        super(Model_SINE_LI_NL, self).__init__(n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, 
                                         flag="SINE", item_norm=item_norm)
        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = temperature
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head
        self.output_units = output_size
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)  # ?,category_num,128
        self.cat_seq = self._concate_multi_seq(self.seq_multi)
        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.item_emb,
                                                            self.the_first_w, self.the_first_bias)  # (?, 128)
        self.user_eb = self.gate_user_eb(self.long_user_embedding, self.cat_seq, self.user_embedding)
        self._xent_loss_weight(self.user_eb, self.seq_multi)
        self.summary_loss()

    def _concate_multi_seq(self, seq_multi):
        splits = [tf.squeeze(seq, axis=1) for seq in tf.split(seq_multi, num_or_size_splits=self.category_num, axis=1)]
        seq_multi_cat = tf.concat(splits, axis=1)
        cat_seq = tf.keras.layers.Dense(self.output_units, activation='relu')(seq_multi_cat)
        return cat_seq
    
    def gate_user_eb(self, long_user_embedding, cat_seq, user_embedding):
        gate_input = concat_func([long_user_embedding, cat_seq, user_embedding])  # (?, 128+128+128)
        gate = tf.keras.layers.Dense(self.output_units, activation='sigmoid')(gate_input)  # (?, 128)
        gate_output = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]) \
            + tf.multiply(1 - x[0], x[2]))([gate, cat_seq, long_user_embedding])  # (?, 128)
        user_eb = self.l2_normalize(gate_output)  # (?, 128)
        return user_eb
        
    def l2_normalize(self, x, axis = -1):
        return tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)

    def attention_level_one(self, user_embedding, item_emb, the_first_w, the_first_bias):
        user_embedding_expanded = tf.expand_dims(user_embedding, axis=1)  # [n,1,128]，转置后变成[n,128,1]
        weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.sigmoid(tf.add(tf.matmul(item_emb, the_first_w), the_first_bias)),
            tf.transpose(user_embedding_expanded, perm=[0,2,1])), perm=[0,2,1]))  # [n,1,20]
        out = tf.reduce_sum(tf.multiply(item_emb, tf.transpose(weight, perm=[0,2,1])), axis=1)  # reduce_sum([n,20,128]multiply[n,20,1])---[n,128]
        return tf.reshape(out, [-1, self.dim])
    
    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(item_emb, self.nbr_mask)#[N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]，意图序列
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)  # D维，预测的下一个意图

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)，不同兴趣的聚合权重
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            return item_emb

    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, dim=1), axis=-1)#[-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = tf.reshape(items, [-1, self.dim])  # N,D
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])  # ?,2,128

        return seq


class Model_SINE_LI_NG(Model):
    def __init__(self, n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head, temperature):
        super(Model_SINE_LI_NG, self).__init__(n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, 
                                         flag="SINE", item_norm=item_norm)
        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = temperature
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head
        self.output_units = output_size
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)  # ?,category_num,128
        self.user_eb_short = self.labeled_attention(self.seq_multi)  # ?*128

        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.item_emb,
                                                            self.the_first_w, self.the_first_bias)  # (?, 128)
        self.user_eb = self.concat_user_eb(self.long_user_embedding, self.user_eb_short)
        self._xent_loss_weight(self.user_eb, self.seq_multi)
        self.summary_loss()

    def concat_user_eb(self, long_user_embedding, user_eb_short):
        cat_user_eb = concat_func([long_user_embedding, user_eb_short])
        user_eb = tf.keras.layers.Dense(self.output_units, activation='relu')(cat_user_eb)  # (?, 128)
        return user_eb
        
    def l2_normalize(self, x, axis = -1):
        return tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)

    def attention_level_one(self, user_embedding, item_emb, the_first_w, the_first_bias):
        user_embedding_expanded = tf.expand_dims(user_embedding, axis=1)  # [n,1,128]，转置后变成[n,128,1]
        weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.sigmoid(tf.add(tf.matmul(item_emb, the_first_w), the_first_bias)),
            tf.transpose(user_embedding_expanded, perm=[0,2,1])), perm=[0,2,1]))  # [n,1,20]
        out = tf.reduce_sum(tf.multiply(item_emb, tf.transpose(weight, perm=[0,2,1])), axis=1)  # reduce_sum([n,20,128]multiply[n,20,1])---[n,128]
        return tf.reshape(out, [-1, self.dim])
    
    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def labeled_attention(self, seq):
        # item_emb = tf.reshape(self.cate_dist, [-1, self.hist_max, self.category_num])
        item_emb = tf.transpose(self.cate_dist, [0, 2, 1])
        item_emb = tf.matmul(item_emb, self.batch_tpt_emb)

        if self.cpt_feat:
            item_emb = item_emb + tf.reshape(self.item_emb, [-1, self.hist_max, self.dim])
        target_item = self.sequence_encode_concept(item_emb, self.nbr_mask)#[N,  D]

        mu_seq = tf.reduce_mean(seq, axis=1)  # [N,H,D] -> [N,D]，意图序列
        target_label = tf.concat([mu_seq, target_item], axis=1)

        mu = tf.layers.dense(target_label, self.dim, name='maha_cpt2', reuse=tf.AUTO_REUSE)  # D维，预测的下一个意图

        wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1)，不同兴趣的聚合权重
        wg = tf.nn.softmax(wg, dim=1)

        user_emb = tf.reduce_sum(seq * wg, axis=1)  # [N,H,D]->[N,D]
        if self.user_norm:
            user_emb = tf.nn.l2_normalize(user_emb, dim=-1)
        return user_emb

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            return item_emb

    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, dim=1), axis=-1)#[-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = tf.reshape(items, [-1, self.dim])  # N,D
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])  # ?,2,128

        return seq


class Model_SINE_LI_NGL(Model):
    def __init__(self, n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, topic_num, category_num, alpha,
                 neg_num, cpt_feat, user_norm, item_norm, cate_norm, n_head, temperature):
        super(Model_SINE_LI_NGL, self).__init__(n_mid, user_count, embedding_dim, hidden_size, output_size, batch_size, seq_len, 
                                         flag="SINE", item_norm=item_norm)
        self.num_topic = topic_num
        self.category_num = category_num
        self.hidden_units = hidden_size
        self.alpha_para = alpha
        self.temperature = temperature
        # self.temperature = 0.1
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.cate_norm = cate_norm
        self.neg_num = neg_num
        self.num_heads = n_head
        self.output_units = output_size
        if cpt_feat == 1:
            self.cpt_feat = True
        else:
            self.cpt_feat = False
        with tf.variable_scope('topic_embed', reuse=tf.AUTO_REUSE):
            self.topic_embed = \
                tf.get_variable(
                    shape=[self.num_topic, self.dim],
                    name='topic_embedding')

        self.seq_multi = self.sequence_encode_cpt(self.item_emb, self.nbr_mask)  # ?,category_num,128
        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.item_emb,
                                                            self.the_first_w, self.the_first_bias)  # (?, 128)
        self.user_eb = self._concate_multi_seq(self.seq_multi, self.long_user_embedding)
        self._xent_loss_weight(self.user_eb, self.seq_multi)
        self.summary_loss()

    def _concate_multi_seq(self, seq_multi, long_user_embedding):
        splits = [tf.squeeze(seq, axis=1) for seq in tf.split(seq_multi, num_or_size_splits=self.category_num, axis=1)]
        splits.append(long_user_embedding)
        seq_multi_cat = tf.concat(splits, axis=1)
        user_eb = tf.keras.layers.Dense(self.output_units, activation='relu')(seq_multi_cat)
        return user_eb
    
    def gate_user_eb(self, long_user_embedding, user_eb_short, user_embedding):
        gate_input = concat_func([long_user_embedding, user_eb_short, user_embedding])  # (?, 128+128+128)
        gate = tf.keras.layers.Dense(self.output_units, activation='sigmoid')(gate_input)  # (?, 128)
        gate_output = tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]) \
            + tf.multiply(1 - x[0], x[2]))([gate, user_eb_short, long_user_embedding])  # (?, 128)
        user_eb = self.l2_normalize(gate_output)  # (?, 128)
        return user_eb
        
    def l2_normalize(self, x, axis = -1):
        return tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis))(x)

    def attention_level_one(self, user_embedding, item_emb, the_first_w, the_first_bias):
        user_embedding_expanded = tf.expand_dims(user_embedding, axis=1)  # [n,1,128]，转置后变成[n,128,1]
        weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.sigmoid(tf.add(tf.matmul(item_emb, the_first_w), the_first_bias)),
            tf.transpose(user_embedding_expanded, perm=[0,2,1])), perm=[0,2,1]))  # [n,1,20]
        out = tf.reduce_sum(tf.multiply(item_emb, tf.transpose(weight, perm=[0,2,1])), axis=1)  # reduce_sum([n,20,128]multiply[n,20,1])---[n,128]
        return tf.reshape(out, [-1, self.dim])
    
    def sequence_encode_concept(self, item_emb, nbr_mask):

        item_list_emb = tf.reshape(item_emb, [-1, self.hist_max, self.dim])

        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten_cpt", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w  = tf.layers.dense(item_hidden, self.num_heads, activation=None)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            seq = tf.reshape(item_emb, [-1, self.num_heads, self.dim])
            if self.num_heads != 1:
                mu = tf.reduce_mean(seq, axis=1)
                mu = tf.layers.dense(mu, self.dim, name='maha_cpt')
                wg = tf.matmul(seq, tf.expand_dims(mu, axis=-1))
                wg = tf.nn.softmax(wg, dim=1)
                seq = tf.reduce_mean(seq * wg, axis=1)
            else:
                seq = tf.reshape(seq, [-1, self.dim])
        return seq

    def seq_aggre(self, item_list_emb, nbr_mask):
        num_aggre = 1
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        with tf.variable_scope("self_atten_aggre", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh)
            item_att_w = tf.layers.dense(item_hidden, num_aggre, activation=None)
            item_att_w = tf.transpose(item_att_w, [0, 2, 1])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, num_aggre, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)

            item_att_w = tf.nn.softmax(item_att_w)

            item_emb = tf.matmul(item_att_w, item_list_emb)

            item_emb = tf.reshape(item_emb, [-1, self.dim])

            return item_emb

    def topic_select(self, input_seq):
        seq = tf.reshape(input_seq, [-1, self.hist_max, self.dim])
        seq_emb = self.seq_aggre(seq, self.nbr_mask)
        if self.cate_norm:
            seq_emb = tf.nn.l2_normalize(seq_emb, dim=-1)
            topic_emb = tf.nn.l2_normalize(self.topic_embed, dim=-1)
            topic_logit = tf.matmul(seq_emb, topic_emb, transpose_b=True)
        else:
            topic_logit = tf.matmul(seq_emb, self.topic_embed, transpose_b=True)#[batch_size, topic_num]
        top_logits, top_index = tf.nn.top_k(topic_logit, self.category_num)#two [batch_size, categorty_num] tensors
        top_logits = tf.sigmoid(top_logits)
        return top_logits, top_index

    def seq_cate_dist(self, input_seq):
        #     input_seq [-1, dim]
        top_logit, top_index = self.topic_select(input_seq)
        topic_embed = tf.nn.embedding_lookup(self.topic_embed, top_index)
        self.batch_tpt_emb = tf.nn.embedding_lookup(self.topic_embed, top_index)#[-1, cate_num, dim]
        self.batch_tpt_emb = self.batch_tpt_emb * tf.tile(tf.expand_dims(top_logit, axis=2), [1, 1, self.dim])
        norm_seq = tf.expand_dims(tf.nn.l2_normalize(input_seq, dim=1), axis=-1)#[-1, dim, 1]
        cores = tf.nn.l2_normalize(topic_embed, dim=-1) #[-1, cate_num, dim]
        cores_t = tf.reshape(tf.tile(tf.expand_dims(cores, axis=1), [1, self.hist_max, 1, 1]), [-1, self.category_num, self.dim])
        cate_logits = tf.reshape(tf.matmul(cores_t, norm_seq), [-1, self.category_num]) / self.temperature #[-1, cate_num]
        cate_dist = tf.nn.softmax(cate_logits, dim=-1)
        return cate_dist

    def sequence_encode_cpt(self, items, nbr_mask):
        item_emb_input = tf.reshape(items, [-1, self.dim])  # N,D
        # self.cate_dist = tf.reshape(self.seq_cate_dist(self.item_emb), [-1, self.category_num, self.hist_max])
        self.cate_dist = tf.transpose(tf.reshape(self.seq_cate_dist(item_emb_input), [-1, self.hist_max, self.category_num]), [0, 2, 1])
        item_list_emb = tf.reshape(item_emb_input, [-1, self.hist_max, self.dim])
        item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])

        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.layers.dense(item_list_add_pos, self.hidden_units, activation=tf.nn.tanh, name='fc1')
            item_att_w = tf.layers.dense(item_hidden, self.num_heads * self.category_num, activation=None, name='fc2')

            item_att_w = tf.transpose(item_att_w, [0, 2, 1]) #[batch_size, category_num*num_head, hist_max]

            item_att_w = tf.reshape(item_att_w, [-1, self.category_num, self.num_heads, self.hist_max]) #[batch_size, category_num, num_head, hist_max]

            category_mask_tile = tf.tile(tf.expand_dims(self.cate_dist, axis=2), [1, 1, self.num_heads, 1]) #[batch_size, category_num, num_head, hist_max]
            # paddings = tf.ones_like(category_mask_tile) * (-2 ** 32 + 1)
            seq_att_w = tf.reshape(tf.multiply(item_att_w, category_mask_tile), [-1, self.category_num * self.num_heads, self.hist_max])

            atten_mask = tf.tile(tf.expand_dims(nbr_mask, axis=1), [1, self.category_num * self.num_heads, 1])

            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            seq_att_w = tf.where(tf.equal(atten_mask, 0), paddings, seq_att_w)
            seq_att_w = tf.reshape(seq_att_w, [-1, self.category_num, self.num_heads, self.hist_max])

            seq_att_w = tf.nn.softmax(seq_att_w)

            # here use item_list_emb or item_list_add_pos, that is a question
            item_emb = tf.matmul(seq_att_w, tf.tile(tf.expand_dims(item_list_emb, axis=1), [1, self.category_num, 1, 1])) #[batch_size, category_num, num_head, dim]

            category_embedding_mat = tf.reshape(item_emb, [-1, self.num_heads, self.dim]) #[batch_size, category_num, dim]
            if self.num_heads != 1:
                mu = tf.reduce_mean(category_embedding_mat, axis=1)  # [N,H,D]->[N,D]
                mu = tf.layers.dense(mu, self.dim, name='maha')
                wg = tf.matmul(category_embedding_mat, tf.expand_dims(mu, axis=-1))  # (H,D)x(D,1) = [N,H,1]
                wg = tf.nn.softmax(wg, dim=1)  # [N,H,1]

                # seq = tf.reduce_mean(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
                seq = tf.reduce_sum(category_embedding_mat * wg, axis=1)  # [N,H,D]->[N,D]
            else:
                seq = category_embedding_mat
            self.category_embedding_mat = seq
            seq = tf.reshape(seq, [-1, self.category_num, self.dim])  # ?,2,128

        return seq


def get_param_var(name, shape, partitioner=None, initializer=None,
                  reuse=None, scope='param'):
    with tf.variable_scope(scope, partitioner=partitioner, reuse=reuse):
        # noinspection PyUnresolvedReferences
        var = tf.get_variable(
            name, shape=shape, dtype=tf.float32,
            initializer=(initializer or tf.contrib.layers.xavier_initializer()),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
    return var


def get_emb_initializer(emb_sz):
    # initialize emb this way so that the norm is around one
    return tf.random_normal_initializer(mean=0.0, stddev=float(emb_sz ** -0.5))
