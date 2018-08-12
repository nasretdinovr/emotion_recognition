"""
A set of classes for logging in multiprocessing environments. The logging system in this case includes two entities:
server and queued logger, which are connected with a multiprocessing queue. Server is represented by LoggingServer
class and a queued logger is represented by QueuedLogger class. In this model, a pair of connected server and queue
share the same name and can be accessed with get_server() and get_logger() functions.
QueuedLogger is a subclass of standard logging.Logger class and from the user side can be used as a usual logger.
The handlers and formatters are set on the server side via set_server_logger_initializer() function. An attempt to add
formatters or handlers to QueuedLogger instance will cause NotImplementedError.

The simplest workflow is:
1. Write a logger_initializer() function which sets up logging system from standard Python library and returns a logger 
   which will handle all messages for a server;
2. In the main process, call the function set_server_logger_initializer(logger_initializer=logger_initializer). It will
   create a server and a queued logger, if they do not exist already.
3. Create subclass of multiprocessing.Process. In the constructor, create an attribute for the logger: 
   self.mp_logger = get_logger(). It is important to do it in the parent process, before you call start() function!
4. Start the server: start_server().
5. Start your process and use self.mp_logger as if it is a standard logger!

Notes:
- You can call start_server() before or after you child process is started or even before self.mp_logger = get_logger().
  But it is preferable to start the server as early as possible so that the queue does not get filled.
- You can use multiple servers (with different loggers). To do this, just pass the logger name (a string) to the
  corresponding functions.
- You should consider calling set_server_init_params() and set_logger_init_params() functions before you set up
  anything.
- The servers are stopped when the main process finishes. Make sure that you have joined you subprocesses or you may
  lose log messages or get deadlocks.
"""

import atexit
import logging
import time
from multiprocessing import Queue, Lock
from Queue import Empty, Full
from ..concurrency.abstract_process_worker import AbstractProcessWorker



DEFAULT_NAME = "default"


def set_server_init_params(**kwargs):
    """
    Set initialization parameters for all new LoggingServer instances.

    :param kwargs: initialization arguments. See LoggingServer constructor.
    """

    _manager.set_server_init_params(**kwargs)


def set_logger_init_params(**kwargs):
    """
    Set initialization parameters for all new QueuedLogger instances.

    :param kwargs: initialization arguments. See QueuedLogger constructor.
    """

    _manager.set_logger_init_params(**kwargs)


def get_server(server_name=DEFAULT_NAME):
    """
    Get the logging server with the name specified by server_name. If it does not exist, new server and a corresponding
    logger are created.
    
    :param server_name: name of the server.
    :return: an instance of LoggingServer.
    """

    return _manager.get_server(server_name)


def get_logger(logger_name=DEFAULT_NAME):
    """
    Get the logger with the name specified by logger_name. If it does not exist, new logger and a corresponding server
    are created.
    
    :param logger_name: name of the queued logger.
    :return: an instance of QueuedLogger.
    """

    return _manager.get_logger(logger_name)


def set_server_logger_initializer(server_name=DEFAULT_NAME, logger_initializer=None):
    """
    Set the initializer for the internal logger of the server. The internal logger is the "regular" logger from the
    standard logging module. It is used to process all LogRecords which were passed through the queue to the server. 
    
    :param server_name: name of the server.
    :param logger_initializer: a callable returning a logging.Logger instance (and performing other standard logging
        setup operation, e.g.handlers and formatters setting). 
    """

    server = _manager.get_server(server_name)
    server.logger_initializer = logger_initializer


def start_server(server_name=DEFAULT_NAME):
    """
    Start a server with a given name. If it does not exist, new server and a corresponding logger are created.
    Use set_server_init_params() to set initialization parameters.
    
    :param server_name: 
    """

    server = _manager.get_server(server_name)
    server.start()


def stop_server(server_name=DEFAULT_NAME):
    """
    Stop a server with a given name. If it does not exist, it will be created.
    Use set_server_init_params() to set initialization parameters.
    
    :param server_name: name of the server to stop
    """

    server = _manager.get_server(server_name)
    server.stop()


def stop_all_servers():
    """
    Stop all logging servers. May cause deadlock if some child processes are writing logs to queued loggers.
    This function is called when the main process finishes, but it is NOT called when the program is killed by a signal
    not handled by Python, when a Python fatal internal error is detected, or when os._exit() is called.
    See: https://docs.python.org/2.7/library/atexit.html 
    """

    _manager.stop_all_servers()


# Stop all the loggers when the main process finishes
# Note that this will not work if the program is killed by a signal not handled by Python, when a Python fatal internal
# error is detected, or when os._exit() is called.
# See: https://docs.python.org/2.7/library/atexit.html
atexit.register(stop_all_servers)


class Manager(object):
    """
    The internal class that aggregates all instances of QueuedLogger and LoggingServer.
    Generally, there is no need to create instances of this class outside this module.
    """

    def __init__(self):
        """
        Initialize new manager.
        """

        self.servers_and_loggers = {}
        self.server_init_params = {}
        self.logger_init_params = {}

    def set_server_init_params(self, **kwargs):
        """
        Set initialization parameters for all new LoggingServer instances.
        
        :param kwargs: initialization arguments. See LoggingServer constructor.
        """

        self.server_init_params = kwargs

    def set_logger_init_params(self, **kwargs):
        """
        Set initialization parameters for all new QueuedLogger instances.
        
        :param kwargs: initialization arguments. See QueuedLogger constructor.
        """

        self.logger_init_params = kwargs

    def get_server(self, name):
        """
        Get LoggingServer with a specified name. If it does not exist, new server and a corresponding logger are
        created.
        
        :param name: name of the logging server.
        :return: LoggingServer instance 
        """

        if name not in self.servers_and_loggers:
            self._new_server_and_logger(name)
        return self.servers_and_loggers[name][0]

    def get_logger(self, name):
        """
        Get QueuedLogger with a specified name. If it does not exist, new server and a corresponding logger are created.
        
        :param name: name of the queued logger
        :return: QueuedLogger instance
        """

        if name not in self.servers_and_loggers:
            self._new_server_and_logger(name)
        return self.servers_and_loggers[name][1]

    def stop_all_servers(self):
        """
        Stop all logging servers. May cause deadlock if some child processes are writing logs to queued loggers.
        """

        for server, __ in self.servers_and_loggers.values():
            server.stop()

    def _new_server_and_logger(self, name):
        new_server = LoggingServer(name, **self.server_init_params)
        new_logger = QueuedLogger(name, **self.logger_init_params)
        new_logger.queue = new_server.queue
        self.servers_and_loggers[name] = (new_server, new_logger)


_manager = Manager()


class QueuedLogger(logging.getLoggerClass()):
    """
    The proxy class. Its instances behave like logging.Logger objects, but all it does is forwarding LogRecords to
    the queue (and thus to the server).
    """

    def __init__(self, name, level=logging.NOTSET, handle_timeout=None):
        """
        A "client-side" logger which forwards all log records to the corresponding server.
        Be careful if handle_timeout is None, the call may block. If the corresponding server is not running
        (or crashed) a deadlock may occur.
        
        :param name: name of this logger (and the corresponding LoggingServer instance) in the module internal 
            data structure.
        :param level: log level (see docs for logging standard module)
        :param handle_timeout: timeout for the logger-server queue. If not None, the message may be lost
        """

        super(QueuedLogger, self).__init__(name, level)
        self.handle_timeout = handle_timeout
        self.queue = None

    def handle(self, record):
        try:
            self.queue.put(record, timeout=self.handle_timeout)
        except (Full, AttributeError):
            pass  # TODO If needed, add a callback (optional)

    def addHandler(self, hdlr):
        raise NotImplementedError

    def removeHandler(self, hdlr):
        raise NotImplementedError

    def callHandlers(self, record):
        raise NotImplementedError

    def getChild(self, suffix):
        raise NotImplementedError


class LoggingServer(AbstractProcessWorker):
    """
    This class aggregates all LogRecords from different processes and moves them to a single logger,
    If the logger was not initialized, the server will "swallow" the messages without outputting them.
    """

    def __init__(self, name, qsize=0, queue_timeout=0.1, logger_initializer=None):
        """
        Create a new logging server. 
        
        :param name: name of the logging server (and corresponding QueuedLogger instance) in the module internal 
            data structure.
        :param qsize: size of the input queue.
        :param queue_timeout: queue query timeout. It is not very important, 0.1 second is a reasonable value for 
            most cases
        :param logger_initializer: a callable objects that sets up logging and returns an instance of logging.Logger,
            that is used to write logs
        """

        # super(AbstractProcessWorker, self).__init__()
        AbstractProcessWorker.__init__(self)
        self.queue = Queue(qsize)
        self.name = name  # By the way it overrides the multiprocessing.Process.name
        self.logger_initializer = logger_initializer
        self._logger = None
        self._lock = Lock()
        self._lock_is_mine = False
        self._queue_timeout = queue_timeout

    def on_start(self):
        if not self._lock.acquire(block=False):
            raise RuntimeError("Only one LoggingServer should run at the same time!")
        self._lock_is_mine = True
        if self.logger_initializer is not None:
            self._logger = self.logger_initializer()
        if self._logger is not None:
            self._logger.info("Logging server '{}' started. PID: {}".format(self.name, self.pid))

    def do_work_once(self):
        try:
            log_record = self.queue.get(timeout=self._queue_timeout)
        except Empty:
            return

        # Forward the log_record to the internal _logger
        if self._logger is not None:
            self._logger.handle(log_record)

    def on_finish(self):
        # Release the lock
        if self._lock_is_mine:  # Any process can release a "foreign" lock so we have a flag
            self._lock_is_mine = False
            self._lock.release()
        if self._logger is not None:
            self._logger.info("Logging server '{}' finished. PID: {}".format(self.name, self.pid))

    def on_crash(self):
        if self._logger is not None:
            self._logger.exception("Logging server '{}' crashed. PID: {}".format(self.name, self.pid))

    def stop(self):
        """
        Stop the logging server if it runs. Does nothing, is the server is not running.
        """

        # Wait until the queue is empty
        while not self.queue.empty():
            time.sleep(0.1)

        # Shutdown the server (someone may put something to a queue - it is not our problem)
        self.deactivate()
        try:
            self.join()
        except AssertionError:
            pass
