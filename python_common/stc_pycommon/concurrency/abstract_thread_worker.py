import abc
from base_abstract_worker import BaseAbstractWorker
import threading


# We do not rely on atomicity of the basic operations
# http://blog.qqrs.us/blog/2016/05/01/which-python-operations-are-atomic/
class AtomicVar(object):
    def __init__(self, init_val=None):
        super(AtomicVar, self).__init__()
        self._val = init_val
        self._lock = threading.Lock()

    def set(self, val):
        with self._lock:
            self._val = val

    def get(self):
        with self._lock:
            val = self._val
        return val


class AbstractThreadWorker(BaseAbstractWorker, threading.Thread):
    """
    Base worker class which implements "multithreading" execution model.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(AbstractThreadWorker, self).__init__()
        self._active = AtomicVar(True)  # It could be False but we set it to True for backward compatibility

    def deactivate(self):
        self._active.set(False)

    def is_active(self):
        return self._active.get()

    def run(self):
        self._active.set(True)
        self._run()
