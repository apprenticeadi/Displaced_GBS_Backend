import sys
import logging
import datetime
import os


class LogUtils:

    @staticmethod
    def log_config(dir=None, module_name='', level=logging.INFO):
        time_stamp = datetime.datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
        if dir is None:
            dir = r'..\Results\logs'
        logging_filename = dir + r'\{}.txt'.format(time_stamp)
        os.makedirs(os.path.dirname(logging_filename), exist_ok=True)

        stdout_handler = logging.StreamHandler(sys.stdout)

        logging.basicConfig(filename=logging_filename, level=level,
                            format='%(levelname)s %(asctime)s %(message)s')

        # make logger print to console (it will not if multithreaded)
        logging.getLogger(module_name).addHandler(stdout_handler)



