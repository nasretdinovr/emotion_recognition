import abc


class AbstractDataLoader:
    """Base class for data loaders for pyTorch nets"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        """Everything that we need to init"""
        pass

    @abc.abstractmethod
    def next_batch(self):
        """
        Returns next batch from training data.
        Make sure your function returns (batch_num, data, labels)
        """
        pass

    @abc.abstractmethod
    def next_val_batch(self):
        """
        Returns next batch from validation data.
        Make sure your function returns (is_finished, data, labels)
        """
        pass
