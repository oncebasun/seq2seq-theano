
import logging
import numpy
import os
import sys
import time
import pickle as cPickle

from contextlib import closing
from six.moves import cPickle
from sampling import SamplingBase

from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)


class SaveLoadUtils(object):
    """Utility class for checkpointing."""

    @property
    def path_to_folder(self):
        return self.folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')
        
    @property
    def path_to_best_parameters(self):
        return os.path.join(self.folder, 'best_params.npz')

    @property
    def path_to_embeddings(self):
        return os.path.join(self.embeddings, 'best_params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        return os.path.join(self.folder, 'log')

    def load_emb_hack(self):
      emb_dict = {}
      with closing(numpy.load('/mounts/Users/cisintern/huiming/universal-mri/Code/_FINAL_ST_MODELS/t1_high/task2/Ens_1_model_german_100_100/best_params.npz')) as source:
        param_values = {}
        for name, value in source.items():
          if name == '-bidirectionalencoder-embeddings.W':
            name_ = name.replace(BRICK_DELIMITER, '/')
            if not name_.startswith('/'):
              name_ = '/' + name_
            new_embs = value
      #print(list(new_embs)[0])
      #print(len(list(new_embs)[0]))
      #exit()
    
      src_voc = cPickle.load(open('/mounts/Users/cisintern/huiming/universal-mri/Data/_ST_FINAL/t1_high/german_src_voc_task2.pkl', 'rb'))
      for sth, idx in src_voc.iteritems():
        emb_dict[sth] = list(new_embs)[idx]

      #print(emb_dict)
      #exit()
      return emb_dict
      

    def load_word2vec(self, path):
      # HACK!:
      path = '/mounts/Users/cisintern/huiming/universal-mri/Data/corpora/for_embeddings/german_embs_SSKIP.100'
      word2vec = {}
      f=open(path, 'r')
      for line in f:    
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])
      #print(word2vec.keys())
      #exit()
      return word2vec

    def load_embeddings_from_text(self, emb_path, src_voc_path, emb_size=300):
      # 1) load a dictionary with the embeddings from a file
      emb_dict = self.load_word2vec(emb_path)
      #emb_dict = {}
      #emb_dict = self.load_emb_hack()

      # 2) match those with the source vocabulary, using (1,0,0,0,0,...) for unknown ones
      src_voc = cPickle.load(open(src_voc_path, 'rb'))
      W = []
      for i in range(3):
        W.append([1] + [0] * (emb_size-1))      
      for char in sorted(src_voc, key=src_voc.get, reverse=False):
        #print char, src_voc[char]
        if char in emb_dict:
          #print(char)
          W.append(emb_dict[char])
        else:
          W.append([1] + [0] * (emb_size-1))

      # 3) convert it into a numpy array
      W = numpy.asarray(W)
      W = W.astype(numpy.float32, copy=False)
      #print(W)
      #exit()

      # 4) return the array
      return W
  

    def load_parameter_values(self, path):
        with closing(numpy.load(path)) as source:
            param_values = {}
            for name, value in source.items():
                if name != 'pkl':
                    name_ = name.replace(BRICK_DELIMITER, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    param_values[name_] = value

        return param_values
        
    '''
    def load_embedding_values(self, path):
        with closing(numpy.load(path)) as source:
            param_values = {}
            
            for name, value in source.items():
                if name != 'pkl' and name == '-decoder-sequencegenerator-readout-lookupfeedbackwmt15-lookuptable.W':
                    print(name)
                    name_ = name.replace(BRICK_DELIMITER, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    
                    param_values[name_] = value

        # /decoder/sequencegenerator/readout/lookupfeedbackwmt15/lookuptable.W
        return param_values
    '''
    def save_parameter_values(self, param_values, path):
        param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
        numpy.savez(path, **param_values)


class CheckpointNMT(SimpleExtension, SaveLoadUtils):
    """Redefines checkpointing for NMT.

        Saves only parameters (npz), iteration state (pickle) and log (pickle).

    """

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        kwargs.setdefault("after_training", True)
        super(CheckpointNMT, self).__init__(**kwargs)

    def dump_parameters(self, main_loop):
        params_to_save = main_loop.model.get_parameter_values()
        self.save_parameter_values(params_to_save,
                                   self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state, self.path_to_iteration_state)

    def dump_log(self, main_loop):
        secure_dump(main_loop.log, self.path_to_log, cPickle.dump)

    def dump(self, main_loop):
        if not os.path.exists(self.path_to_folder):
            os.mkdir(self.path_to_folder)
        print("")
        logger.info(" Saving model")
        start = time.time()
        logger.info(" ...saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" ...saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" ...saving log")
        self.dump_log(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def do(self, callback_name, *args):
        try:
            self.dump(self.main_loop)
        except Exception:
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (self.path_to_folder +
                                                            'params.npz',))
                                                            
class LoadEmbeddings(TrainingExtension, SaveLoadUtils):
    """Loads pretrained embeddings."""

    def __init__(self, embedding_path, src_voc, **kwargs):
        self.embeddings = embedding_path 
        self.src_voc = src_voc
        super(LoadEmbeddings, self).__init__(**kwargs)

    def before_training(self):
        if not os.path.exists(self.embeddings):
            logger.info("No embeddings found")
            return
        logger.info("Loading embeddings into the main loop")
        try:
            self.load_to(self.main_loop)
        except Exception:
            reraise_as("Failed to load embeddings")
        
    def load_embeddings(self):
        logger.info("Loading pretrained embeddings")
        return self.load_embedding_values(self.embeddings)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Loading pretrained embeddings")
        try:
            params_this = main_loop.model.get_parameter_dict()
            new_embs = self.load_embeddings_from_text(self.path_to_embeddings, self.src_voc)

            if params_this['/bidirectionalencoder/embeddings.W'].get_value().shape != new_embs.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))
            else:
              logger.info(
                " Embedding shape fits!")
            params_this['/bidirectionalencoder/embeddings.W'].set_value(new_embs)
            logger.info(
                " Embeddings loaded successfully.")
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
            

class LoadNMT(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadNMT, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = main_loop.model.get_parameter_dict()
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))
            logger.info(
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
            
 
class LoadOnlyBestModel(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadOnlyBestModel, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the best state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        #print('would load best')
        #exit()
        return self.load_parameter_values(self.path_to_best_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = main_loop.model.get_parameter_dict()
            #for key, value in params_this.iteritems():
            #  print(key + ': ' + value)
            #print(params_this)
            #sys.exit(0)
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))
            logger.info(
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        '''
        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        
        
        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        '''
        
                   
class LoadOnlyModel(TrainingExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadOnlyModel, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop, get_embs = False):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = main_loop.model.get_parameter_dict()
            #for key, value in params_this.iteritems():
            #  print(key + ': ' + value)
            #print(params_this)
            #sys.exit(0)
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    if get_embs:
                      print(pname)
                      if pname == '/bidirectionalencoder/embeddings.W':
                        print(pname + ':')
                        print(val)
                        outfile = open('/mounts/Users/cisintern/huiming/character_embeddings/embs_It_Sp', 'w')
                        for line in range(len(val)):
                          for column in range(len(val[line])):
                            outfile.write(str((val[line][column])) + ' ')
                          outfile.write('\n')
                        exit()
                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))
            logger.info(
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        '''
        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        
        
        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        '''
        
        
class LoadOnlyModel_later(SimpleExtension, SaveLoadUtils):
    """Loads parameters log and iterations state."""

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        self.every_n_batches=1,
        super(LoadOnlyModel_later, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        print('doing LoadOnlyModel_later')
        #sys.exit(0)
        if not os.path.exists(self.path_to_folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.path_to_folder))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path_to_folder
        except Exception:
            reraise_as("Failed to load the state")

    def load_parameters(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        logger.info(" Reloading model")
        try:
            logger.info(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = main_loop.model.get_parameter_dict()
            #for key, value in params_this.iteritems():
            #  print(key + ': ' + value)
            #print(params_this)
            #sys.exit(0)
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logger.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    params_this[pname].set_value(val)
                    logger.info(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))
            logger.info(
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

        '''
        try:
            logger.info(" Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        
        
        try:
            logger.info(" Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))
        '''
