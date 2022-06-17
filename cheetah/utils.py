from cheetah.params import BATCH_SIZE
from tensorflow.data import AUTOTUNE
from time import time


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        duration = t2 - t1
        print(f'Function {func.__name__!r} executed in {duration:.4f}s')
        return result, duration
    return wrap_func
