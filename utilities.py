from scipy import signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE_FREQUENCY = 1000

def make_plot_sign_changes(emg_data, variability_data, filename):
    """Creates a plot presenting voltage and torque"""
    fig, axs = plt.subplots(2)
    plt.style.use('seaborn-whitegrid')
    axs[0].plot(emg_data, linewidth=1)
    axs[0].set_xlim([0, len(emg_data)])
    axs[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axs[0].set_title("sEMG Signal", fontsize=18)
    axs[0].set_ylabel("sEMG = [mV]", fontsize=16)
    axs[0].set_xlabel("t = [ms]", fontsize=16)
    axs[1].plot(variability_data, linewidth=1, color="red")
    axs[1].set_ylim([0, max(variability_data)+10])
    axs[1].set_xlim([0, len(variability_data)])
    axs[1].set_title("Sign changes", fontsize=18)
    axs[1].set_ylabel("variability", fontsize=16)
    axs[1].set_xlabel("t = [ms]", fontsize=16)
    fig = plt.gcf()
    fig.set_size_inches(24, 12)
    plt.setp(axs, xticks=[i for i in range(0, len(emg_data) + 100, 100)])
    plt.setp(axs[0].get_xticklabels(), rotation=50, fontsize=14)
    plt.setp(axs[0].get_yticklabels(), fontsize=14)
    plt.setp(axs[1].get_xticklabels(), rotation=50, fontsize=14)
    plt.setp(axs[1].get_yticklabels(), fontsize=14)
    # extent = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.tight_layout(pad=2)
    plt.savefig('./figures/{0}.svg'.format(filename), format='svg', dpi=300, bbox_inches='tight')

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