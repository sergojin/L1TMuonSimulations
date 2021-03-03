import logging

# Copied from https://docs.python.org/2/howto/logging.html

def getLogger():
  # create logger
  logger = logging.getLogger('test9')
  logger.setLevel(logging.DEBUG)

  # create file handler which logs even debug messages
  fh = logging.FileHandler('test9.log')
  fh.setLevel(logging.DEBUG)

  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  # create formatter and add it to the handlers
  #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s')
  fh.setFormatter(formatter)
  formatter = logging.Formatter('[%(levelname)-8s] %(message)s')
  ch.setFormatter(formatter)

  # add the handlers to the logger
  if not len(logger.handlers):
    logger.addHandler(fh)
    logger.addHandler(ch)

  return logger
