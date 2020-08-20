from utils import * 
import pickle 
import time 
from tqdm import tqdm
import argparse
import numpy as np 
import pickle 
import tensorflow as tf 
from tensorflow.contrib import learn 
from tflearn.data_utils import to_categorical, pad_sequences

from loguru import logger

class URLNet:
    def __init__(self, data_dir: str, subword_dict_dir: str, word_dict_dir: str,
                 checkpoint_dir: str, char_dict_dir: str,
                 labelling: str = "multiclass", max_len_words: int = 200,
                 max_len_chars: int = 200, max_len_subwords: int = 20, delimit_mode: int = 1,
                 emb_dim: int = 32, emb_mode: int = 1, batch_size: int = 128):

        logger.debug("Initializing model")

        self.checkpoint_dir = checkpoint_dir

        self.emb_mode = emb_mode
        self.emb_dim = emb_dim
        self.delimit_mode = delimit_mode

        self.max_len_chars = max_len_chars
        self.max_len_words = max_len_words
        self.max_len_subwords = max_len_subwords

        self.batch_size = batch_size
        
        urls, self.labels, original_labels = read_data(data_dir, labelling)

        x, word_reverse_dict = get_word_vocab(urls, max_len_words) 
        word_x = get_words(x, word_reverse_dict, delimit_mode, urls) 

        ngram_dict = pickle.load(open(subword_dict_dir, "rb")) 
        logger.info("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
        word_dict = pickle.load(open(word_dict_dir, "rb"))
        logger.info("size of word vocabulary (train): {}".format(len(word_dict)))
        self.ngramed_id_x, self.worded_id_x = ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict) 
        chars_dict = pickle.load(open(char_dict_dir, "rb"))          
        self.chared_id_x = char_id_x(urls, chars_dict, max_len_chars)    

        logger.info("Number of testing urls: {}".format(len(self.labels)))

    def test_step(self, x, sess):
        p = 1.0
        if self.emb_mode == 1: 
            feed_dict = {
                self.input_x_char_seq: x[0],
                self.dropout_keep_prob: p}  
        elif self.emb_mode == 2: 
            feed_dict = {
                self.input_x_word: x[0],
                self.dropout_keep_prob: p}
        elif self.emb_mode == 3: 
            feed_dict = {
                self.input_x_char_seq: x[0],
                self.input_x_word: x[1],
                self.dropout_keep_prob: p}
        elif self.emb_mode == 4: 
            feed_dict = {
                self.input_x_word: x[0],
                self.input_x_char: x[1],
                self.input_x_char_pad_idx: x[2],
                self.dropout_keep_prob: p}
        elif self.emb_mode == 5:  
            feed_dict = {
                self.input_x_char_seq: x[0],
                self.input_x_word: x[1],
                self.input_x_char: x[2],
                self.input_x_char_pad_idx: x[3],
                self.dropout_keep_prob: p}
        preds, s = sess.run([self.predictions, self.scores], feed_dict)
        return preds, s

    def predict(self):
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        graph = tf.Graph() 
        with graph.as_default(): 
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth=True 
            sess = tf.Session(config=session_conf)
            with sess.as_default(): 
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file) 
                
                if self.emb_mode in [1, 3, 5]: 
                    self.input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
                if self.emb_mode in [2, 3, 4, 5]:
                    self.input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
                if self.emb_mode in [4, 5]:
                    self.input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
                    self.input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]
                
                if self.emb_mode == 1: 
                    batches = batch_iter(list(self.chared_id_x), self.batch_size, 1, shuffle=False) 
                elif self.emb_mode == 2: 
                    batches = batch_iter(list(self.worded_id_x), self.batch_size, 1, shuffle=False) 
                elif self.emb_mode == 3: 
                    batches = batch_iter(list(zip(self.chared_id_x, self.worded_id_x)), self.batch_size, 1, shuffle=False)
                elif self.emb_mode == 4: 
                    batches = batch_iter(list(zip(self.ngramed_id_x, self.worded_id_x)), self.batch_size, 1, shuffle=False)
                elif self.emb_mode == 5: 
                    batches = batch_iter(list(zip(self.ngramed_id_x, self.worded_id_x, self.chared_id_x)), self.batch_size, 1, shuffle=False)    
                all_predictions = []
                all_scores = []
                
                nb_batches = int(len(self.labels) / self.batch_size)
                if len(self.labels) % self.batch_size != 0: 
                    nb_batches += 1 
                print("Number of batches in total: {}".format(nb_batches))
                it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} test_size {}".format(self.emb_mode, self.delimit_mode, len(self.labels)), ncols=0)
                for idx in it:
                #for batch in batches:
                    batch = next(batches)

                    if self.emb_mode == 1: 
                        x_char_seq = batch 
                    elif self.emb_mode == 2: 
                        x_word = batch 
                    elif self.emb_mode == 3: 
                        x_char_seq, x_word = zip(*batch) 
                    elif self.emb_mode == 4: 
                        x_char, x_word = zip(*batch)
                    elif self.emb_mode == 5: 
                        x_char, x_word, x_char_seq = zip(*batch)            

                    x_batch = []    
                    if self.emb_mode in[1, 3, 5]: 
                        x_char_seq = pad_seq_in_word(x_char_seq, self.max_len_chars) 
                        x_batch.append(x_char_seq)
                    if self.emb_mode in [2, 3, 4, 5]:
                        x_word = pad_seq_in_word(x_word, self.max_len_words) 
                        x_batch.append(x_word)
                    if self.emb_mode in [4, 5]:
                        x_char, x_char_pad_idx = pad_seq(x_char, self.max_len_words, self.max_len_subwords, self.emb_dim)
                        x_batch.extend([x_char, x_char_pad_idx])
                    
                    batch_predictions, batch_scores = self.test_step(x_batch, sess)            
                    all_predictions = np.concatenate([all_predictions, batch_predictions]) 
                    all_scores.extend(batch_scores)

                    it.set_postfix()

        if self.labels is not None: 
            correct_preds = float(sum(all_predictions == self.labels)) 
            print("Accuracy: {}".format(correct_preds/float(len(self.labels))))
        
        return all_predictions, all_scores