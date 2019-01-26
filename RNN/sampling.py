# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import io
import logging
import numpy
import operator
import os
import sys
import re
import signal
import theano
import time
from time import gmtime, strftime

import cPickle

from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.search import BeamSearch
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.evaluators import AggregationBuffer, DatasetEvaluator
from stream import _ensure_special_tokens

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)

class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_word(self, seq, ivocab):
        #print(ivocab)
        #sys.exit(0)
        return u" ".join([ivocab.get(idx, u"<UNK>") for idx in seq])


class AttentionMonitoring(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, decoder, encoder, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
                 #trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(AttentionMonitoring, self).__init__(**kwargs)
        self.decoder= decoder
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()
        theinputs = [inp for inp in self.model.inputs]
        encodedStates = encoder.representation
        self.testFct = theano.function(theinputs, encodedStates)
        #self.attention_fct = theano.function([theinputs], self.decoder.attention.weighted_averages, givens = {states: encodedStates})

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)
            
        #print(self.trg_vocab)
        #sys.exit(0)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]


        
        # Sample
        #print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            
            _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T
 
            #weighted_averages, weights_T = self.attention.take_glimpses(inp[None, :], self.attention.state_names)
            
            weighted_averages = self.testFct(inp[None, :])
            
            print('myTest:')
            print(weighted_averages)
            print(weighted_averages.shape)
            #print('\nweights.T')
            #print(weights_T)
            sys.exit(0)
            

class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
                 #trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)
            
        #print(self.trg_vocab)
        #sys.exit(0)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        #print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            
            _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print(u"Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab))
            print("Input in indices: ", input_[i][:input_length])
            print("Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_word(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample in indices: ", outputs[:sample_length])
            print("Sample cost: ", costs[:sample_length].sum())
            print()

class Predictor(SimpleExtension, SamplingBase):
#class Predictor(TrainingExtension):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
                 #trg_ivocab=None, src_vocab_size=None, **kwargs):
       
        super(Predictor, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()
        self.input_word = None
        self.target_word = None
        self.predicted_word = None
        self.resulting_predictions = {}
        self.before_first_epoch=True
        
        
    #@application(inputs=["input1"], outputs=["output"])
    def apply(self, input1):
        a0 = self.input_projection.apply(input1)
        a1 = self.rnn.apply(a0)
        a2 = self.output_projection.apply(a1)
        return a2

    def dos(self, which_callback, *args):
      return None
       
    def do(self, which_callback, *args):
        print('starting prediction')
        sys.exit(0)
        
        self.hook_samples = 20000 # originally 10
        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)
            
        #print(self.trg_vocab)
        #sys.exit(0)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        print('batch_size:')
        print(batch_size)
        
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        counter = 0
        finishedAll = True
        outfile = io.open('testtesttest', 'w', encoding='utf-8')
        for i in range(hook_samples):
            counter += 1
            print("Counter: " + str(counter))
            if not finishedAll:
              print("something went wrong")
              sys.exit(0) 
            finishedAll = False
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            
            _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            # NOTE: There are two types of lost words. One is because the label has not been seen in the training data (rare).
            # The other reason is unknown. However, the word is lost at this point already. 
            self.input_word = self._idx_to_word(input_[i][:input_length], self.src_ivocab)
            self.target_word = self._idx_to_word(target_[i][:target_length], self.trg_ivocab)
            self.predicted_word = self._idx_to_word(outputs[:sample_length], self.trg_ivocab)
            #outfile.write(u"Input : " + self.input_word)
            print(u"Input : ", self.input_word)
            print("Input in indices: ", input_[i][:input_length])
            print("Target: ", self.target_word)
            print("Sample: ", self.predicted_word)
            print("Sample in indices: ", outputs[:sample_length])
            print("Sample cost: ", costs[:sample_length].sum())
            print()
            
            help_word_input = u''
            help_word_result = u''
            help_tag = u''
            help_tag = self.input_word.split(' ')[1] + u''
            for letter in self.input_word.split(' ')[2:len(self.input_word.split(' '))-1]:
              help_word_input = u'' + help_word_input + letter
            for letter in self.predicted_word.split(' ')[1:len(self.predicted_word.split(' '))-1]:
              help_word_result = u'' + help_word_result + letter
            self.resulting_predictions[(help_tag, help_word_input)] = help_word_result
            
            finishedAll = True
        #print('resulting_predictions:')
        #print(resulting_predictions)
        cPickle.dump(self.resulting_predictions, open('intermediate_results.pkl', 'wb'))
        print('prediction done')
        #sys.exit(0)
     
   
# TODO: use (to be done) dev stream for this
# THIS IS NOT WORKING
class ShowTestVariable(SimpleExtension, SamplingBase):
    """Stores the best model based on accuracy on the dev set."""

    def __init__(self, model, data_stream,
                 config, trg_ivocab=None, **kwargs):
        super(ShowTestVariable, self).__init__(**kwargs)
        self.config = config
        self.data_stream = data_stream
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):
        print('\nlalalalala test\n')
        #exit()
        
        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, 1)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        #print()
        for i in range(1):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            
            readouts, output = (self.sampling_fn(inp[None, :]))
            
            print(readouts)
            print(output)
        exit()
        
        
        
# TODO: use (to be done) dev stream for this
class AccuracyValidator(SimpleExtension, SamplingBase):
    """Stores the best model based on accuracy on the dev set."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1, trg_ivocab=None, **kwargs):
        super(AccuracyValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.best_accuracy = 0.0
        self.best_epoch = -1
        self.validation_results_file = self.config['saveto'] + '/validation/accuracies_' + self.config['lang']
        #self.verbose = config.get('val_set_out', None) # TODO: set this to a file and True for a sentence output

        # Helpers
        self.vocab = None
        self.src_vocab = None
        self.src_ivocab = None
        self.trg_vocab = None
        self.trg_ivocab = None
        # The next two are hacks.
        self.unk_sym = '<UNK>'
        self.eos_sym = '</S>'
        self.unk_idx = None
        self.eos_idx = None
        self.sampling_fn = self.model.get_theano_function()
        
        #self.beam_search = BeamSearch(samples=samples)
        self.eow_idx = 2 # TODO: this is a hack
       
        # Create saving directory if it does not exist
        validation_path = self.config['saveto'] + '/validation/'
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        #if self.config['reload']:
        if False:
            try:
                acc_file = open(validation_path + 'accuracy_score')
                for line in acc_file: # well, this should be only one line
                  self.best_accuracy = float(line.strip()) 
                logger.info("Accuracy reloaded")
            except:
                logger.info("No former accuracy found")
           
        res_file = open(self.validation_results_file, 'a')
        res_file.write(str(time.time()) + '\n')
        res_file.close()

    def do(self, which_callback, *args):
        '''
        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return
        '''
        
        # Evaluate and save if necessary
        no_epochs_done = self.main_loop.status['epochs_done']
        #print(no_epochs_done)
        #sys.exit(0)
        self._evaluate_model(no_epochs_done)
        #self._save_model(self._evaluate_model(), no_epochs_done)

    def _evaluate_model(self, no_epochs_done):
        logger.info("Started Validation.")
        val_start_time = time.time()
        error_count = 0
        total_count = 0

        # Get target vocabulary
        if not self.trg_ivocab:
            # Load dictionaries and ensure special tokens exist
            self.src_vocab = _ensure_special_tokens(
                cPickle.load(open(self.config['src_vocab'])),
                bos_idx=0, eos_idx=2, unk_idx=1)
            self.trg_vocab = _ensure_special_tokens(
                cPickle.load(open(self.config['trg_vocab'])),
                bos_idx=0, eos_idx=2, unk_idx=1)
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.unk_idx = self.src_vocab[self.unk_sym]
            self.eos_idx = self.src_vocab[self.eos_sym]
      
        '''
        print('length data_stream:')
        print(len(self.data_stream))
        sys.exit(0)
        '''
        ana_file = io.open(self.validation_results_file + '_det', 'w', encoding='utf-8') # this should always contain the last
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """
            #print(line)
            #sys.exit(0)
            seq = line[0] #self._oov_to_unk(line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (1, 1))
            
            seq2 = line[2]
            target_ = numpy.tile(seq2, (1, 1))
            
            #print('target')
            #print(target_)
            
            #print('input')
            #print(input_)
            
            batch_size = input_.shape[0]
            #print('batch_size:')
            #print(batch_size)
            #sys.exit(0)
            
            #src_batch = anIter[main_loop.data_stream.mask_sources[0]]
            #input_ = src_batch[:, :]
        
        
            for j in range(batch_size):
        
              input_length = get_true_length(input_[j], self.src_vocab)
              target_length = get_true_length(target_[j], self.trg_vocab)
        
              inp = input_[j, :input_length]
              _1, outputs, _2, _3, costs, _4 = (self.sampling_fn(inp[None, :]))
              outputs = outputs.flatten()
              sample_length = get_true_length(outputs, self.trg_vocab)
        
              # NOTE: There are two types of lost words. One is because the label has not been seen in the training data (rare).
              # The other reason is unknown. However, the word is lost at this point already. 
              input_word = _idx_to_word(input_[j][:input_length], self.src_ivocab)
              #print(input_word)
              target_word = _idx_to_word(target_[j][:target_length], self.trg_ivocab)
              #print(target_word)
              predicted_word = _idx_to_word(outputs[:sample_length], self.trg_ivocab)
              #print(predicted_word)
              #sys.exit(0)
              # TODO: get target word
              ana_file.write(input_word + ' / ' + target_word + '\n')
              ana_file.write(predicted_word + '\n\n')
              if target_word != predicted_word:
                error_count += 1
              total_count += 1

        new_accuracy = (total_count - error_count)*1.0 / total_count

        self._save_model(new_accuracy, no_epochs_done)
        
        res_file = open(self.validation_results_file, 'a')
        res_file.write(str(no_epochs_done) + '\t' + str(new_accuracy) + '\t' + str(error_count) + '\t' + str(total_count) + '\t(Best: ' + str(self.best_epoch) + ')\n')
        res_file.close()

        ana_file.close()
        logger.info("Validation finished. Current accuracy on dev set: " + str(new_accuracy))
        #return new_accuracy

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, accuracy, no_epochs_done):
        #print('test...lalala')
        #exit()
        if accuracy > self.best_accuracy or (no_epochs_done == 5 and self.best_epoch == -1):
            self.best_accuracy = accuracy
            self.best_epoch = no_epochs_done
            #print('**Accuracy:**')
            #print(accuracy)
            print('New highscore!\nSaving the model... \n')
            #sys.exit(0)
            # Accuracy instead of BLEU score
            model = ModelInfo(accuracy, self.config['saveto'])

            ''' just store it, no saving of old ones... hahah
            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))
            '''
            
            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
             
            param_values = self.main_loop.model.get_parameter_values()
            param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
            numpy.savez(model.path, **param_values)
            with open(self.config['saveto'] + '/best_params.lg', 'a') as log_file:
              log_file.write(strftime("%Y-%m-%d %H:%M:%S"))
              log_file.write('\nBest params stored with validation score of ' + str(self.best_accuracy) + '\n')
            #numpy.savez(
            #    model.path, **self.main_loop.model.get_parameter_dict())
            #numpy.savez(
            #    os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
            #    bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)
        else:
          if (no_epochs_done - self.best_epoch >= 35) and (no_epochs_done >= 100) and False: # do NO early stopping
          #if (self.best_epoch != -1 and no_epochs_done - self.best_epoch > 30) or (self.best_epoch == -1 and no_epochs_done == 50):
            res_file = open(self.validation_results_file, 'a')
            res_file.write('Early stopping here...\n')
            res_file.write('Best accuracy on dev: ' + str(self.best_accuracy) + ' [epoch ' + str(self.best_epoch) + ']\n')
            res_file.close()
            self.main_loop.stop()
                

class BleuValidator(SimpleExtension, SamplingBase):
    # TODO: a lot has been changed in NMT, sync respectively
    """Implements early stopping based on BLEU score."""

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1, trg_ivocab=None,
                 normalize=True, **kwargs):
        # TODO: change config structure
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.verbose = config.get('val_set_out', None) # TODO: set this to a file and True for a sentence output

        # Helpers
        '''
        self.vocab = data_stream.dataset.dictionary
        self.trg_ivocab = trg_ivocab
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_grndtruth'], '<']
        '''
        self.beam_search = BeamSearch(samples=samples)
        self.eow_idx = 2 # TODO: this is a hack
       
        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        #if self.config['reload']:
        if False: 
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        print('BLEU1')
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        # Get target vocabulary
        if not self.trg_ivocab:
            sources = self._get_attr_rec(self.main_loop, 'data_stream')
            trg_vocab = sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        if self.verbose:
            ftrans = open(self.config['val_set_out'], 'w')

        
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = line[0] #self._oov_to_unk(
                #line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))
            
            print('input')
            print(input_)
            #sys.exit(0)
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out, self.trg_ivocab)

                except ValueError:
                    logger.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()

        logger.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logger.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logger.info(bleu_score)
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if bleu_score < 0:
        #if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            numpy.savez(
                model.path, **self.main_loop.model.get_parameter_dict())
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_params.npz')
        return gen_path
      
def get_true_length(seq, vocab):
    try:
        return seq.tolist().index(vocab['</S>']) + 1
    except ValueError:
        return len(seq)
            
def _idx_to_word(seq, ivocab):
    #print(ivocab)
    #sys.exit(0)
    return u" ".join([ivocab.get(idx, u"<UNK>") for idx in seq])
