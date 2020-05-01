#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader model"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

from .config import override_model_args
from .rnn_reader import RnnDocReader
from .docqa import DocQA
from .lang_disc import LanguageDetector
from .utils import freeze_net, unfreeze_net

logger = logging.getLogger(__name__)


class DocReader(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def set_iterators (self, train_loader_target,
                 train_loader_source_Q,
                 train_loader_target_Q):
        self.train_loader_target = train_loader_target
        self.train_loader_source_Q = train_loader_source_Q
        self.train_loader_target_Q = train_loader_target_Q

        # Make Iterators for Data Loaders
        self.train_iter_source_Q = iter (self.train_loader_source_Q)
        self.train_iter_target = iter (self.train_loader_target)
        self.train_iter_target_Q = iter (self.train_loader_target_Q)

    def __init__(self, args, word_dict, feature_dict, state_dict=None, normalize=True):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        

        # Building network. If normalize is false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        # 1. Transformer ( Feature Extracter )
        if args.model_type == 'rnn':
            self.F = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # 2. Question - Answering network
        self.P = DocQA (doc_hidden_size, question_hidden_size, normalize)

        # 3. Language Detector
        self.Q = LanguageDetector(2, doc_hidden_size, args.dropoutQ, True)
        # self.Q = LanguageDetector (doc_hidden_size, question_hidden_size, normalize, 2, args.dropoutQ, True)

        # TODO : Write good comment
        self.sum_source_q, self.sum_target_q = dict(), dict()

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict['F']:
                fixed_embedding = state_dict['F'].pop('fixed_embedding')
                self.F.load_state_dict(state_dict['F'])
                self.F.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.F.load_state_dict(state_dict['F'])
            self.P.load_state_dict(state_dict['P'])
            self.Q.load_state_dict(state_dict['Q'])

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.F.embedding.weight.data
            self.F.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.F.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.F.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            # Skip first line if of form count/dim.
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.F.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.F.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            freeze_net (self.F.embedding)
 
        # Optimizer for F and P
        parametersF = [p for p in self.F.parameters() if p.requires_grad]
        parametersP = [p for p in self.P.parameters() if p.requires_grad]
        parameters = parametersF + parametersP
        if self.args.optimizer == 'sgd':
            self.optimizerF = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizerF = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

        # Optimizer for Q
        self.optimizerQ = optim.Adam (self.Q.parameters(), lr = self.args.Q_learning_rate)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def getDataPoint (self, ex):
        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
            target_s = ex[5].cuda(non_blocking=True)
            target_e = ex[6].cuda(non_blocking=True)
        else:
            inputs = [e if e is None else e for e in ex[:5]]
            target_s = ex[5]
            target_e = ex[6]

        return {
            'inputs' : inputs,
            'target_s' : target_s,
            'target_e' : target_e
        }

    def getInputForQ(self, a, b):
        b = b.float()
        b = b.unsqueeze(1)
        a = b @ a
        a = a.squeeze(1)
        return a

    def update(self, ex, n_critic, current_epoch):
        """Forward a batch of examples; step the optimizer to update weights."""
        if (not self.optimizerF) or (not self.optimizerQ):
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.F.train()
        self.P.train()
        self.Q.train()

        data_point_source =  self.getDataPoint (ex)

        try:
            inputs_target = next (self.train_iter_target)
        except:
            self.train_iter_target = iter (self.train_loader_target)
            inputs_target = next (self.train_iter_target)

        data_point_target = self.getDataPoint (inputs_target)

        # Q Iterations
        freeze_net (self.F)
        freeze_net (self.P)
        unfreeze_net (self.Q)

        for _ in range (n_critic):
            # clip Q weights
            for p in self.Q.parameters():
                p.data.clamp_ (self.args.clip_lower, self.args.clip_upper)
            self.Q.zero_grad ()
            
            # Get a minibatch of data
            try:
                inputs_source_q = next (self.train_iter_source_Q)
            except StopIteration:
                self.train_iter_source_Q = iter (self.train_loader_source_Q)
                inputs_source_q = next (self.train_iter_source_Q)
            
            try:
                inputs_target_q = next (self.train_iter_target_Q)
            except StopIteration:
                self.train_iter_target_Q = iter (self.train_loader_target_Q)
                inputs_target_q = next (self.train_iter_target_Q)

            q_inputs_source = self.getDataPoint (inputs_source_q)
            q_inputs_target = self.getDataPoint (inputs_target_q)

            features_source = self.F(*(q_inputs_source['inputs']))
            input_q = self.getInputForQ (features_source[0], features_source[2])
            o_source_ad = self.Q (input_q)
            l_source_ad = torch.mean (o_source_ad)
            (-l_source_ad).backward()
            logger.debug (f'Q grad norm: {self.Q.net[1].weight.grad.data.norm()}')
            if self.sum_source_q.get (current_epoch, None) is None:
                self.sum_source_q[current_epoch] = [0, 0.0]
            self.sum_source_q[current_epoch][0] += 1
            self.sum_source_q[current_epoch][1] += l_source_ad.item()

            features_target = self.F(*(q_inputs_target['inputs']))
            input_q = self.getInputForQ (features_target[0], features_target[2])
            o_target_ad = self.Q (input_q)
            l_target_ad = torch.mean (o_target_ad)
            l_target_ad.backward()
            logger.debug (f'Q grad norm: {self.Q.net[1].weight.grad.data.norm()}')
            if self.sum_target_q.get (current_epoch, None) is None:
                self.sum_target_q[current_epoch] = [0, 0.0]
            self.sum_target_q[current_epoch][0] += 1
            self.sum_target_q[current_epoch][1] += l_target_ad.item()

            self.optimizerQ.step()

        # F and P iteration
        unfreeze_net (self.F)
        unfreeze_net (self.P)
        freeze_net (self.Q)

        # Clip Q weights
        for p in self.Q.parameters():
            p.data.clamp_(self.args.clip_lower, self.args.clip_upper)
        self.F.zero_grad()
        self.P.zero_grad()

        features_source = self.F(*(data_point_source['inputs']))
        score_s, score_e = self.P(*features_source)

        # Compute loss and accuracies
        l_source_qa = F.nll_loss(score_s, data_point_source['target_s']) + F.nll_loss(score_e, data_point_source['target_e'])
        l_source_qa.backward(retain_graph=True)
        
        input_q = self.getInputForQ (features_source[0], features_source[2])
        o_source_ad = self.Q(input_q)
        l_source_ad = torch.mean(o_source_ad)
        (self.args.lambd * l_source_ad).backward(retain_graph=True)

        features_target = self.F(*(data_point_target['inputs']))
        input_q = self.getInputForQ (features_target[0], features_target[2])
        o_target_ad = self.Q(input_q)
        l_target_ad = torch.mean(o_target_ad)
        (-self.args.lambd * l_target_ad).backward(retain_graph=True)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.F.parameters(),
                                       self.args.grad_clipping)
        torch.nn.utils.clip_grad_norm_(self.P.parameters(),
                                        self.args.grad_clipping)

        self.optimizerF.step ()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return l_source_qa.item(), ex[0].size(0)

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.F.module.embedding.weight.data
                fixed_embedding = self.F.module.fixed_embedding
            else:
                embedding = self.F.embedding.weight.data
                fixed_embedding = self.F.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.F.eval()
        self.P.eval()

        # Transfer to GPU
        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
        else:
            inputs = [e for e in ex[:5]]

        # Run forward
        with torch.no_grad():
            features = self.F(*inputs)
            score_s, score_e = self.P(*features)

        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode_candidates, args)
            else:
                return self.decode_candidates(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.

        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e. Except only consider
        spans that are in the candidates list.
        """
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            # Extract original tokens stored with candidates
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                # try getting from globals? (multiprocessing in pipeline mode)
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')

            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    # Match! Record its score.
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)

            if len(scores) == 0:
                # No candidates present
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                # Rank found candidates
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            networkF = self.F.module
            networkP = self.P.module
            networkQ = self.Q.module
        else:
            networkF = self.F
            networkP = self.P
            networkQ = self.Q
        state_dict = dict()
        state_dict['F'] = copy.copy(networkF.state_dict())
        state_dict['P'] = copy.copy(networkP.state_dict())
        state_dict['Q'] = copy.copy(networkQ.state_dict())

        if 'fixed_embedding' in state_dict['F']:
            state_dict['F'].pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            networkF = self.F.module
            networkP = self.P.module
            networkQ = self.Q.module
        else:
            networkF = self.F
            networkP = self.P
            networkQ = self.Q

        state_dict = dict()
        state_dict['F'] = networkF.state_dict()
        state_dict['P'] = networkP.state_dict()
        state_dict['Q'] = networkQ.state_dict()
        optimizer = {'F' : self.optimizerF.state_dict(), 'Q' : self.optimizerQ.state_dict()}
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': optimizer,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, args, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return DocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.F = self.F.cuda()
        self.P = self.P.cuda()
        self.Q = self.Q.cuda()

    def cpu(self):
        self.use_cuda = False
        self.F = self.F.cpu()
        self.P = self.P.cpu()
        self.Q = self.Q.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.F = torch.nn.DataParallel(self.F)
        self.P = torch.nn.DataParallel(self.P)
        self.Q = torch.nn.DataParallel(self.Q)

