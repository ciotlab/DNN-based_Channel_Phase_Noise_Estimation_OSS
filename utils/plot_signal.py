import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from dataset import get_dataset_and_dataloader


def plot_signal(signal_dict, shape=None):
    num_plot = len(signal_dict)
    if shape is None:
        shape = (1, num_plot)
    fig = plt.figure(figsize=(shape[1]*5, shape[0]*4))
    gs = GridSpec(shape[0], shape[1])
    for it, (k, v) in enumerate(list(signal_dict.items())):
        name = k
        plot_type = v['type']  # reim (real+imag), real, imag, magnitude, phase, scalar
        data = v['data']
        if 'x' in v:
            x = v['x']
        else:
            x_range = v.get('x_range', (0, data.shape[-1]))
            x = np.arange(x_range[0], x_range[1])
            data = np.transpose(data[..., x])
        ax = fig.add_subplot(gs[it])
        ax.set_title(name)
        if plot_type == 'reim':
            ax.plot(x, np.real(data))
            ax.plot(x, np.imag(data))
        elif plot_type == 'real':
            ax.plot(x, np.real(data))
        elif plot_type == 'imag':
            ax.plot(x, np.imag(data))
        elif plot_type == 'magnitude':
            ax.plot(x, np.absolute(data))
        elif plot_type == 'phase':
            ax.plot(x, np.angle(data))
        elif plot_type == 'scalar':
            ax.plot(x, data)
    return fig


if __name__ == "__main__":
    # Dataset
    data_file_name = 'channel_mmWave.mat'
    batch_size = 32
    snr_db = 60
    mod_order = 16
    ref_conf_dict = {'dmrs': (0, 3072, 1), 'ptrs': (6, 3072, 48)}
    num_guard_subcarriers = 1024
    rnd_seed = 0
    num_workers = 0
    dataset, dataloader = get_dataset_and_dataloader(data_file_name=data_file_name, batch_size=batch_size, snr_db=snr_db,
                                                     mod_order=mod_order, ref_conf_dict=ref_conf_dict, num_guard_subcarriers=num_guard_subcarriers,
                                                     rnd_seed=rnd_seed, num_workers=num_workers, is_phase_noise=True, is_channel=True, is_noise=True)

    it = iter(dataloader)
    data = next(it)
    sig_dict = {'pn_time_real': {'data': data['pn_time'], 'type': 'real'},
                'pn_time_imag': {'data': data['pn_time'], 'type': 'imag'},
                'ch_freq_real': {'data': data['ch_freq'], 'type': 'real'},
                'ch_freq_imag': {'data': data['ch_freq'], 'type': 'imag'},
                'ch_time_real': {'data': data['ch_time'], 'type': 'real'},
                'ch_time_imag': {'data': data['ch_time'], 'type': 'imag'}}
    f = plot_signal(sig_dict, shape=(3, 2))
    f.show()


