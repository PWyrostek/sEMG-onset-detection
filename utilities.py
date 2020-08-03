from scipy import signal
from sklearn.decomposition import PCA

BASE_FREQUENCY = 1000


def estimate_theta_0(data, M):
    sum = 0
    for i in range(0, M):
        sum += data[i] ** 2
    return sum / M

def butterworth_filter(data, cutoff_frequency):
    """Applies butterworth filter with given cutoff frequency to the data"""
    w = cutoff_frequency / (BASE_FREQUENCY / 2)
    b, a = signal.butter(6, w, btype='lowpass', analog=False)
    data = signal.filtfilt(b, a, data)
    return data


def whitening(data):
    """TODO: Whitening filter"""
    pca = PCA(whiten=True)
    whitened = pca.fit_transform(data.reshape(-1, 1))
    return whitened