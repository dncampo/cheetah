from tensorflow.data import AUTOTUNE

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size=16)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
