import os


class Logger():
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)

        self.logger = open(fname, 'w+', encoding='utf-8')

    def log(self, string, file_only = False):
        if not file_only:
            print(string)
        self.logger.write(string+'\n')
        self.logger.flush()

    def close(self):
        self.logger.close()

LOGGER = dict()

def get_logger(path = ""):
    global LOGGER
    try:
        logger = LOGGER[path]
    except:
        logger = Logger(path)
        LOGGER[path] = logger
    
    return logger