import numpy as np

#CC executes z-score normalization of obs and passes them on via GLOBAL_OBS_STAT
class RunningObsStat:
    def __init__(self, shape, eps=1e-8):
        self.eps = eps
        self.n   = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2   = np.zeros(shape, dtype=np.float32)

    def update(self, x: np.ndarray):
        x = x.astype(np.float32, copy=False)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    def std(self):
        # unbiased var for n>1
        var = self.M2 / max(self.n - 1, 1)
        #return np.sqrt(var + self.eps, dtype=np.float32)
        return np.sqrt(var + self.eps).astype(np.float32, copy=False)

    def normalize(self, x, clip=5.0):

        import numpy as np
        x = np.asarray(x, dtype=np.float32)
        std = self.std()
        z = (x - self.mean) / std
        if clip is not None:
            z = np.clip(z, -clip, clip)
        return z


GLOBAL_OBS_STAT = RunningObsStat(shape=(32,))