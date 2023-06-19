import numpy as np
import pickle
import random

def flatten_list(lst: list):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

def center_data(x: np.ndarray):
    return (x - x.mean(axis=0)) / x.std(axis=0)

