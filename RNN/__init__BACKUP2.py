#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, io
import sys
import cPickle
import math
import numpy
from operator import mul

from collections import Counter
from stream import _ensure_special_tokens
from theano import tensor
import theano
from toolz import merge
import blocks
from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.bricks.cost import MisclassificationRate
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant, Identity
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.select import Selector

from RNN.checkpoint import CheckpointNMT, LoadNMT, LoadOnlyModel, LoadOnlyBestModel, LoadOnlyModel_later
from RNN.model import BidirectionalEncoder, Decoder
from RNN.sampling import AccuracyValidator, AttentionMonitoring, Sampler, Predictor, ShowTestVariable

try:
    from blocks_extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(config, tr_stream, dev_stream, use_bokeh=False, the_task=None, the_track=None):

    config['the_task'] = the_task
    # Create Theano variables
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        # end_embed is dimension of word embedding matrix in encoder; enc_nhids number of hidden units in encoder GRU
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2, config['use_attention'], cost_type=config['error_fct'])
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)
    testVar = decoder.getTestVar(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)
   
    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    my_rng = numpy.random.RandomState(config['rng_value']) 
    if config['identity_init']:
      encoder.weights_init = decoder.weights_init = Identity()
    else:
      encoder.weights_init = decoder.weights_init = IsotropicGaussian(
          config['weight_scale'])
      encoder.rng = decoder.rng = my_rng
    
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    encoder.bidir.prototype.rng = my_rng
    decoder.transition.weights_init = Orthogonal()
    decoder.transition.rng = my_rng
    encoder.initialize()
    decoder.initialize()

    # apply dropout for regularization
    if config['dropout'] < 1.0:
        # dropout is applied to the output of maxout in ghog
        logger.info('Applying dropout')
        dropout_inputs = [x for x in cg.intermediary_variables
                          if x.name == 'maxout_apply_output']
        cg = apply_dropout(cg, dropout_inputs, config['dropout'])

    # Apply weight noise for regularization
    if config['weight_noise_ff'] > 0.0:
        logger.info('Applying weight noise to ff layers')
        enc_params = Selector(encoder.lookup).get_params().values()
        enc_params += Selector(encoder.fwd_fork).get_params().values()
        enc_params += Selector(encoder.back_fork).get_params().values()
        dec_params = Selector(
            decoder.sequence_generator.readout).get_params().values()
        dec_params += Selector(
            decoder.sequence_generator.fork).get_params().values()
        dec_params += Selector(decoder.state_init).get_params().values()
        cg = apply_noise(cg, enc_params+dec_params, config['weight_noise_ff'], seed=my_rng)

    cost = cg.outputs[0]

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))
    


    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))


    # Set up training model
    logger.info("Building model")
    training_model = Model(cost)    

    # Set extensions
    logger.info("Initializing extensions")
    # this is ugly code and done, because I am not sure if the order of the extensions is important
    if 'track2' in config['saveto']: # less epochs for track 2, because of more data
      if config['early_stopping']:
	extensions = [
	    FinishAfter(after_n_epochs=config['finish_after']/2),
	    #FinishAfter(after_n_batches=config['finish_after']),
	    TrainingDataMonitoring([cost], after_batch=True),
	    Printing(after_batch=True),
	    CheckpointNMT(config['saveto'],
			  every_n_batches=config['save_freq'])
	]
      else:
	extensions = [
	    FinishAfter(after_n_epochs=config['finish_after']/2),
	    #FinishAfter(after_n_batches=config['finish_after']),
	    TrainingDataMonitoring([cost], after_batch=True),
	    Printing(after_batch=True),
	    CheckpointNMT(config['saveto'],
			  every_n_batches=config['save_freq'])
	]
    else:
      if config['early_stopping']:
	extensions = [
	    FinishAfter(after_n_epochs=config['finish_after']),
	    #FinishAfter(after_n_batches=config['finish_after']),
	    TrainingDataMonitoring([cost], after_batch=True),
	    Printing(after_batch=True),
	    CheckpointNMT(config['saveto'],
			  every_n_batches=config['save_freq'])
	]
      else:
	extensions = [
	    FinishAfter(after_n_epochs=config['finish_after']),
	    #FinishAfter(after_n_batches=config['finish_after']),
	    TrainingDataMonitoring([cost], after_batch=True),
	    Printing(after_batch=True),
	    CheckpointNMT(config['saveto'],
			  every_n_batches=config['save_freq'])
	]

    # Set up beam search and sampling computation graphs if necessary
    if config['hook_samples'] >= 1:
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs

    
    # Add sampling
    if config['hook_samples'] >= 1:
        logger.info("Building sampler")
        extensions.append(
            Sampler(model=search_model, data_stream=tr_stream,
                    hook_samples=config['hook_samples'],
                    #every_n_batches=1,
                    every_n_batches=config['sampling_freq'],
                    src_vocab_size=8))
                    #src_vocab_size=config['src_vocab_size']))
    
    # Add early stopping based on bleu
    if config['val_set'] is not None:
        logger.info("Building accuracy validator")
        extensions.append(
            AccuracyValidator(sampling_input, samples=samples, config=config,
                          model=search_model, data_stream=dev_stream,
                          after_training=True,
                          #after_epoch=True))
                          every_n_epochs=5))
    else:
        logger.info("No validation set given for this language")
    
    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto']))
       
    
    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )
    
    # Train!
    main_loop.run()
    

def mainPredict(config, data_to_predict_stream, use_ensemble, lang=None, et_version=False, use_bokeh=False, the_track=None):
    # Create Theano variables
    assert the_track != None
    
    logger.info('Creating theano variables')
    source_sentence = tensor.lmatrix('source')
    source_sentence_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')
    sampling_input = tensor.lmatrix('input')

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])
    decoder = Decoder(
        config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
        config['enc_nhids'] * 2, cost_type=config['error_fct'])
    cost = decoder.cost(
        encoder.apply(source_sentence, source_sentence_mask),
        source_sentence_mask, target_sentence, target_sentence_mask)

    logger.info('Creating computational graph')
    cg = ComputationGraph(cost)

    # Initialize model
    logger.info('Initializing model')
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(
        config['weight_scale'])
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
    logger.info("Total number of parameters: {}".format(len(shapes)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                               Selector(decoder).get_parameters())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.items():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info("Total number of parameters: {}"
                .format(len(enc_dec_param_dict)))
    
    
    # Set extensions
    logger.info("Initializing (empty) extensions")
    extensions = [
    ]

    logger.info("Building sampling model")
    sampling_representation = encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))
    generated = decoder.generate(sampling_input, sampling_representation)
    search_model = Model(generated)

    _, samples = VariableFilter(
        bricks=[decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is next_outputs
    
    # Reload the model (as this is prediction, it is 100% necessary):
    if config['reload']:
        #extensions.append(LoadOnlyBestModel(config['saveto'])) # without early stopping use LoadOnlyModel here!
        extensions.append(LoadOnlyModel(config['saveto'])) # without early stopping use LoadOnlyModel here!
    else:
        raise Exception('No model available for prediction! (Check config[\'reload\'] variable)')

    
    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )

    # Initialize main loop
    logger.info("Initializing main loop")
    main_loop = MainLoop(
        model=search_model,
        algorithm=algorithm,
        #algorithm=None,
        data_stream=data_to_predict_stream,
        extensions=extensions
    )

    predictByHand(main_loop, decoder, data_to_predict_stream, use_ensemble, lang, et_version, config, the_track=the_track)

# This now additionally stores a dictionary from solutions to probability in the format (input, output) -> prob.
def predictByHand(main_loop, decoder, data_to_predict_stream, use_ensemble, lang, et_version, config, monitor_attention=False, the_track=None):
    resulting_predictions = {}
    pred_with_prob = {} # for the bootstrapping
    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    bin_model = main_loop.model
    sources = [inp.name for inp in bin_model.inputs] # = 'input'
    theinputs = [inp for inp in bin_model.inputs]
    decoder_output = [v for v in bin_model.variables if v.name == 'decoder_generate_output_0'][0]
    
    sampling_fn = bin_model.get_theano_function() # the function to call for prediction

    # Load vocabularies and invert if necessary
    # WARNING: Source and target indices from data stream
    #  can be different
    print('*')
    print(config['src_vocab'])
    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        cPickle.load(open(config['src_vocab'])),
        bos_idx=0, eos_idx=2, unk_idx=1)
    trg_vocab = _ensure_special_tokens(
        cPickle.load(open(config['trg_vocab'])),
        bos_idx=0, eos_idx=2, unk_idx=1)
    src_ivocab = {v: k for k, v in src_vocab.items()}
    trg_ivocab = {v: k for k, v in trg_vocab.items()}
    
    epoch_iter = data_to_predict_stream.get_epoch_iterator()

    counter = 0
    
    for anIter in epoch_iter:
      
      input_=anIter[0]
      
      if counter%500 == 0:
        print(counter)
      counter += 1
      
      inp = input_

      the_one = False

      _1, outputs, _2, glimpses, costs, output_probs = (sampling_fn([inp]))
     
      # Combining the probabilities for the sinngle letters to get a total probability.
      prob_word = reduce(mul, output_probs, 1)[0]
      
      outputs = outputs.flatten()
      sample_length = get_true_length(outputs, trg_vocab)
       
      input_word = _idx_to_word(input_, src_ivocab)
      predicted_word = _idx_to_word(outputs[:sample_length], trg_ivocab)
 
      help_word_input = u''
      help_word_result = u''
      help_tag = u''

      resulting_predictions[u' '.join(input_word.split(' ')[1:len(input_word.split(' '))-1])] = u''.join(predicted_word.split(' ')[1:len(predicted_word.split(' '))-1])
      pred_with_prob[(u' '.join(input_word.split(' ')[1:len(input_word.split(' '))-1]), u''.join(predicted_word.split(' ')[1:len(predicted_word.split(' '))-1]))] = prob_word

    results_path = 'results/__'
    if 'paper' in config['saveto']:
      results_path += 'paper_version/'
    else:
      results_path += '__'.join(config['saveto'].split('task')[0].split('/')) + '/track' + str(the_track) + '/'

    if config['the_task'] == 3:
      if et_version:
        cPickle.dump(resulting_predictions, open('ETS_' + lang + '_task3/Ens_' + str(use_ensemble) + '_intermediate_results.pkl', 'wb'))
      else:
        cPickle.dump(resulting_predictions, open(results_path + 'task' + str(config['the_task']) + '/' + lang + '/Ens_' + str(use_ensemble) + '_intermediate_results.pkl', 'wb'))
    else:
      cPickle.dump(resulting_predictions, open(results_path + 'task' + str(config['the_task']) + '/' + lang + '/Ens_' + str(use_ensemble) + '_intermediate_results.pkl', 'wb'))
    print('prediction done')
    print('INFO: storing intermediate results to: ' + results_path + 'task' + str(config['the_task']) + '/' + lang + '/Ens_' + str(use_ensemble) + '_intermediate_results.pkl')

    print('Storing the dictionary with the most likely solutions to ' + config['saveto'] + 'test_sol.pkl')
    cPickle.dump(pred_with_prob, open(config['saveto'] + '/test_sol.pkl', 'wb'))
    print(pred_with_prob)

    
    
def get_true_length(seq, vocab):
    try:
        return seq.tolist().index(vocab['</S>']) + 1
    except ValueError:
        return len(seq)
            
def _idx_to_word(seq, ivocab):
    return u" ".join([ivocab.get(idx, u"<UNK>") for idx in seq])
