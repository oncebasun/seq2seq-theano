def get_config_cs2en():
    config = {}

    # Settings which should be given at start time, but are not, for convenience
    config['the_task'] = 0
    
    # Settings ----------------------------------------------------------------
    config['allTagsSplit'] = 'allTagsSplit/' # can be 'allTagsSplit/', 'POSextra/' or ''
    config['identity_init'] = True
    config['early_stopping'] = False # this has no use for now
    config['use_attention'] = True # if we want attention output at test time; still no effect for training
    
    # Model related -----------------------------------------------------------

    # Definition of the error function; right now only included in baseline_ets
    config['error_fct'] = 'categorical_cross_entropy'

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 300 # orig: 100
    config['dec_nhids'] = 300 # orig: 100
    
    # For the initialization of the parameters.
    config['rng_value'] = 11

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 300 # orig: 300
    config['dec_embed'] = 300 # orig: 300

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'model'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 20

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.5

    # Vocabulary/dataset related ----------------------------------------------

    # Corpus vocabulary pickle file
    config['corpus_data'] = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/Corpora/corpus_voc_'
    
    # Root directory for dataset
    datadir = '/mounts/Users/cisintern/huiming/SIGMORPHON/Code/src/baseline/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    if config['the_task'] > 1:
      config['src_vocab'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '_src_voc_task' + str(config['the_task']) + '.pkl']
      config['trg_vocab'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '_trg_voc_task' + str(config['the_task']) + '.pkl'] # introduce "german" or so here
    else:
      config['src_vocab'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '_src_voc.pkl']
      config['trg_vocab'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '_trg_voc.pkl'] # introduce "german" or so here

    # Source and target datasets
    config['src_data'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '-task' + str(config['the_task']) + '-train_src']
    config['trg_data'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '-task' + str(config['the_task']) + '-train_trg']

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    # This will be read at runtime from a file.
    config['src_vocab_size'] = 159
    config['trg_vocab_size'] = 61

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bow_token'] = '<S>'
    config['eow_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Validation set source file; this is the test file, because there is only a test set for two languages
    config['val_set'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '-task' + str(config['the_task']) + '-test_src']

    # Validation set gold file
    config['val_set_grndtruth'] = ['/mounts/Users/cisintern/huiming/SIGMORPHON/Code/data/forRnn/', '-task' + str(config['the_task']) + '-test_trg']

    # Print validation output to file
    config['output_val_set'] = False

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of epochs
    config['finish_after'] = 500

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 50

    # Show this many samples at each sampling
    config['hook_samples'] = 2

    # Start bleu validation after this many updates
    config['val_burn_in'] = 80000
    
    config['lang'] = None

    return config
