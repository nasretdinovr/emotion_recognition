def list_split(a, n):
    """
    Split a into n parts
    :param a: what to split
    :param n: number of parts to split
    :return: list of n lists
    """
    k, m = len(a) / n, len(a) % n
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)]