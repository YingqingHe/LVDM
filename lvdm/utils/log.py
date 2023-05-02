import os
import logging
import multiprocessing as mproc
import types

#: number of available CPUs on this computer
CPU_COUNT = int(mproc.cpu_count())
#: default date-time format
FORMAT_DATE_TIME = '%Y%m%d-%H%M%S'
#: default logging tile
FILE_LOGS = 'logging.log'
#: default logging template - log location/source for logging to file
STR_LOG_FORMAT = '%(asctime)s:%(levelname)s@%(filename)s:%(processName)s - %(message)s'
#: default logging template - date-time for logging to file
LOG_FILE_FORMAT = logging.Formatter(STR_LOG_FORMAT, datefmt="%H:%M:%S")
#: define all types to be assume list like
ITERABLE_TYPES = (list, tuple, types.GeneratorType)
def release_logger_files():
    """ close all handlers to a file
    >>> release_logger_files()
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    0
    """
    for hl in logging.getLogger().handlers:
        if isinstance(hl, logging.FileHandler):
            hl.close()
            logging.getLogger().removeHandler(hl)

def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file
    :param str path_out: path to the output folder
    :param str file_name: log file name
    :param bool reset: reset all previous logging into a file
    >>> set_experiment_logger('.')
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    1
    >>> release_logger_files()
    >>> os.remove(FILE_LOGS)
    """
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    if reset:
        release_logger_files()
    path_logger = os.path.join(path_out, file_name)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(LOG_FILE_FORMAT)
    log.addHandler(fh)

def set_ptl_logger(path_out, phase, file_name="ptl.log", reset=True):
    """ set the logger to file
    :param str path_out: path to the output folder
    :param str file_name: log file name
    :param bool reset: reset all previous logging into a file
    >>> set_experiment_logger('.')
    >>> len([1 for lh in logging.getLogger().handlers
    ...      if type(lh) is logging.FileHandler])
    1
    >>> release_logger_files()
    >>> os.remove(FILE_LOGS)
    """
    file_name = f"ptl_{phase}.log"
    level = logging.INFO
    log = logging.getLogger("pytorch_lightning")
    log.setLevel(level)

    if reset:
        release_logger_files()
    
    path_logger = os.path.join(path_out, file_name)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(level)
    fh.setFormatter(LOG_FILE_FORMAT)
    
    log.addHandler(fh)
