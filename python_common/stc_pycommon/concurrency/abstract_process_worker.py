import abc
from python_common.stc_pycommon.concurrency.base_abstract_worker import BaseAbstractWorker
from multiprocessing import Process, Value


class AbstractProcessWorker(BaseAbstractWorker, Process):
    """
    Base worker class which implements "multiprocessing" execution model.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(AbstractProcessWorker, self).__init__()
        self._active = Value('i', 1)  # It could be zero but we set it to 1 for backward compatibility

    def deactivate(self):
        with self._active.get_lock():
            self._active.value = 0

    def is_active(self):
        return self._active.value

    def run(self):
        self._active.value = 1
        self._run()
