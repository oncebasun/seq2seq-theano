# use export PYTHONPATH=$PYTHONPATH:/mounts/Users/cisintern/huiming/Programme/pybrain-master

import configurations
import sys
import time
import logging



def main():
  # Load the configurations.
    configuration = getattr(configurations,'')()
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    
  
if __name__ == "__main__":
    main()

