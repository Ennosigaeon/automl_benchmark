import multiprocessing


class NoDaemonProcess(multiprocessing.Process):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        super().__init__(None, target, name, args, kwargs)

    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
