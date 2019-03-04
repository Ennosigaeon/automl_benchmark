import logging


def setup(id: int = None):
    logging.basicConfig(level=40)  # 10: debug; 20: info

    logger = logging.getLogger()
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file = 'application.log'
    if id is not None:
        file = 'application-{}.log'.format(id)

    file_handler = logging.FileHandler(file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get():
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)
    return logger
