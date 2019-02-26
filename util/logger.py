import logging


def setup():
    logging.basicConfig(level=40)  # 10: debug; 20: info

    logger = logging.getLogger()
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('application.log', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get():
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)
    return logger
