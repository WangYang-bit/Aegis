import os, sys
from collections import OrderedDict
import pickle

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

pickle_cache = OrderedDict()
pickle_cache_size = 600

def pickle_cache_load(file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        return pickle_cache[file_path]
    else:
        data = pickle.load(open(file_path, 'rb'))
        if len(pickle_cache) >= pickle_cache_size:
            pickle_cache.popitem(last=False)
        pickle_cache[file_path] = data
        return data
    

def pickle_cache_dump(data, file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        pickle_cache[file_path] = data
    pickle.dump(data, open(file_path, 'wb'))

def main():
    perturbation = {"perturbation" : 1}
    pickle_cache_dump()

if __name__ == "__main__":
    main()