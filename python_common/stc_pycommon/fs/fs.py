import fnmatch
import os
import random
from hashlib import md5
from time import localtime

common_file_extensions = '.jpg', '.png', '.jpeg', '.bmp'


def check_extension(filename, extensions):
    return os.path.splitext(filename)[1].lower() in extensions


def file_is_empty(filepath):
    return os.stat(filepath).st_size == 0


def full_split(path):
    path_components = []
    while True:
        head, tail = os.path.split(path)
        if head != "":
            path_components.insert(0, tail)
        else:
            break
    return path_components


def selective_walk_files(top, topdown=True, onerror=None, followlinks=False, min_depth=None, max_depth=None):
    for dirpath, dirnames, filenames in os.walk(top, topdown=True, onerror=None, followlinks=False):
        # Check depth
        if min_depth is not None or max_depth is not None:
            rel_dirpath = os.path.relpath(dirpath, top)
            cur_depth = len(full_split(rel_dirpath))

            if min_depth is not None and cur_depth < min_depth:
                continue
            if max_depth is not None and cur_depth > max_depth:
                continue

        # Check filter
        yield dirpath, filenames


# TODO rewrite using selective_walk_files()
def find_files(dir_path, extensions=common_file_extensions, save_folders=False):
    if save_folders:
        folder_list = []
        for r, ds, fs in os.walk(dir_path):
            files = [os.path.join(r, fn) for fn in fs if fn.lower().endswith(extensions)]
            if len(files) > 0:
                folder_list.append(files)
        return folder_list
    else:
        return [os.path.join(r, fn)
                for r, ds, fs in os.walk(dir_path)
                for fn in fs if fn.lower().endswith(extensions)]


def clean_dir(dir_path, pattern):
    for root, dirs, files in os.walk(dir_path):
        for file in fnmatch.filter(files, pattern):
            os.remove(os.path.join(root, file))


def random_prefix():
    return md5(str(localtime()) + str(random.random())).hexdigest()


# TODO rewrite using selective_walk_files()
def find_files_pattern(dir_path, pattern):
    list = []
    for root, dirs, files in os.walk(dir_path):
        for file in fnmatch.filter(files, pattern):
            list.append(os.path.join(root, file))
    return list
