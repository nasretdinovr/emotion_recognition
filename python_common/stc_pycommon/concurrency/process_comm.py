"""
The classes for "Future-style" inter-process communication are provided here.
It is needed when multiple processes ("clients") send tasks (or queries) to another process
("server"). The server has an input queue that accepts tasks from clients. The problem is that
the order of the tasks is arbitrary so the single output queue containing the results will not work.
The solution is the Future instance, which is connected with the Task instance with a pipe.
Here is a typical usage sequence:
- The client calls create_future() and gets Task instance and Future instance (connected with a
  pipe).
- The Task instance is passed to the server via multiprocessing.Queue.
- Server gets the task from Task.task attribute, executes it and calls Task.done().
- The client reads the result from Future.wait_and_get().

WARNINGS:

The performance of this method is not very high: it can take a few milliseconds to perform all
aforementioned operations.

This methods are generally NOT thread-safe and should be used for inter-process communication only.

One should be careful to avoid freezes and deadlocks, as the call Task.done() MAY block until 
Future.wait_and_get() is called from the client side. The block will not occur if the result which
is passed to Task.done() is sufficiently small (depending on the platform).
"""


import sys
from multiprocessing import Pipe, reduction


def create_future(task):
    """
    Create new Task and Future instances.
    :param task: the task that should be delivered to the server.
    """
    (conn_in, conn_out) = Pipe(duplex=False)
    return Task(task, conn_out), Future(conn_in)


if sys.platform == 'win32':
    reduce_connection_platform = reduction.reduce_pipe_connection
else:
    reduce_connection_platform = reduction.reduce_connection


class Task(object):
    """
    The Task class (should be used by server).
    """
    def __init__(self, task, conn_out):
        """
        Constructor. Generally, should not be called by the user (use create_future() function).
        """

        super(Task, self).__init__()
        self.task = task
        self._reduced_conn = reduce_connection_platform(conn_out)
        self._done = False

    def done(self, result):
        """
        Send the result of the performed task to the client.
        
        Note that this call may block in some cases (this is platform-dependent behavior).
        In some cases, the receiver will not get the result. Be careful with very large objects as
        they may corrupt the pipe and the client will stay blocked.
        
        :param result: the result of the performed task.
        :raises ValueError for very large results 
                (approximately 32 MB+, though it depends on the OS).
        :raises IOError if the pipe is broken/closed.
        :raises RuntimeError if it has already been called once before.
        """

        if self._done:
            raise RuntimeError("Task.done() can be called only once!")

        # "Unreduce" the connection (make a regular Pipe connection)
        conn_out = self._reduced_conn[0](*self._reduced_conn[1])

        # Send the result
        try:
            conn_out.send(result)
        finally:
            conn_out.close()

        # Switch the flag
        self._done = True


class Future(object):
    """
    The Future class (should be used by client).
    """

    def __init__(self, conn_in):
        """
        Constructor. Generally, should not be called by the user (use create_future() function).
        """

        super(Future, self).__init__()
        self._conn_in = conn_in
        self.read = False

    def wait_and_get(self, timeout=None):
        """
        Wait for the response from the server and get the result.
        
        :param timeout: timeout, in seconds. If None than timeout is infinite.
        :return: the result of the performed task.
        :raises RuntimeError if it has already been called once before.
        :raises IOError if the pipe is broken/closed
        :raises EOFError if the Pipe is closed and there is nothing to read from it
        :raises StopIteration if the timeout passed and nothing was read from the pipe
        """

        if self.read:
            raise RuntimeError("The result cannot be read twice from Future!")

        if self._conn_in.poll(timeout):  # Wait for the data with the timeout
            result = self._conn_in.recv()
            self._conn_in.close()
            self.read = True
            return result
        else:
            raise StopIteration()
