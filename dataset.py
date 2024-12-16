import os
import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ChannelDataset(IterableDataset):
    def __init__(self, data_file_name, batch_size, snr_db, mod_order=4, ref_conf_dict=None, num_guard_subcarriers=1024,
                 rnd_seed=0, is_phase_noise=True, is_channel=True, is_noise=True):
        file_path = Path(__file__).parents[0].resolve() / 'dataset' / data_file_name
        data = h5py.File(file_path)
        self._ch_freq = data['ch_freq']['real'] + 1j * data['ch_freq']['imag']
        self._pn_time = data['pn_time']['real'] + 1j * data['pn_time']['imag']
        self._n_subc = self._ch_freq.shape[1] - num_guard_subcarriers
        self._subc_start_idx = int(np.ceil(num_guard_subcarriers / 2))
        self._n_symb = self._pn_time.shape[1]
        self._batch_size = batch_size
        self._snr_db = snr_db
        self._mod_order = mod_order
        self._is_phase_noise = is_phase_noise
        self._is_channel = is_channel
        self._is_noise = is_noise
        self._base_rnd_seed = rnd_seed
        self._rng = None  # random generator
        self._ref_mask_dmrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask_ptrs = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_mask = np.zeros((self._n_symb, self._n_subc)).astype(bool)
        self._ref_signal = None
        self._set_ref_signal(ref_conf_dict)

    @property
    def n_dmrs(self):
        return int(np.sum(self._ref_mask_dmrs))

    @property
    def n_ptrs(self):
        return int(np.sum(self._ref_mask_ptrs))

    def _set_ref_signal(self, ref_conf_dict):
        self._ref_mask_dmrs[0, np.arange(*ref_conf_dict['dmrs'])] = True
        for s in range(self._n_symb):
            self._ref_mask_ptrs[s, np.arange(*ref_conf_dict['ptrs'])] = True
        self._ref_mask = np.logical_or(self._ref_mask_dmrs, self._ref_mask_ptrs)
        ref_num = int(np.sum(self._ref_mask))
        ref_mod_order = int(np.sqrt(4))  # qpsk
        ref_mod_power_correction = np.sqrt(6 / ((ref_mod_order - 1) * (ref_mod_order + 1)))
        tmp_rng = np.random.default_rng(0)
        ref_mod_index = tmp_rng.integers(low=0, high=ref_mod_order, size=(ref_num, 2))
        self._ref_signal = (ref_mod_index - (ref_mod_order - 1) / 2) * ref_mod_power_correction

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            seed = self._base_rnd_seed
        else:
            worker_id = worker_info.id
            seed = self._base_rnd_seed + worker_id
        self._rng = np.random.default_rng(seed)
        return self

    def __next__(self):
        ch_data_size = self._ch_freq.shape[0]
        ch_freq_idx = self._rng.integers(low=0, high=ch_data_size, size=(self._batch_size,))
        ch_freq = np.fft.fftshift(self._ch_freq[ch_freq_idx, :], axes=(-1,))
        ch_freq = ch_freq[:, self._subc_start_idx: self._subc_start_idx + self._n_subc]
        ch_time = np.fft.ifft(self._ch_freq[ch_freq_idx, :], axis=-1, norm='ortho')
        pn_data_size = self._pn_time.shape[0]
        pn_time_idx = self._rng.integers(low=0, high=pn_data_size, size=(self._batch_size,))
        pn_time = self._pn_time[pn_time_idx, :]
        noise = self._generate_noise()
        tx_signal = self._generate_tx_signal()
        ch_tx_signal = ch_freq[:, np.newaxis, :] * tx_signal if self._is_channel else tx_signal
        rx_signal = ch_tx_signal * pn_time[:, :, np.newaxis] if self._is_phase_noise else ch_tx_signal
        rx_signal = rx_signal + noise if self._is_noise else rx_signal
        ref_tx_signal_dmrs = tx_signal[:, self._ref_mask_dmrs]
        ref_tx_signal_ptrs = tx_signal[:, self._ref_mask_ptrs]
        ref_rx_signal_dmrs = rx_signal[:, self._ref_mask_dmrs]
        ref_rx_signal_ptrs = rx_signal[:, self._ref_mask_ptrs]
        data = {'ref_mask_dmrs': self._ref_mask_dmrs, 'ref_mask_ptrs': self._ref_mask_ptrs,
                'tx_signal': tx_signal, 'rx_signal': rx_signal,
                'ch_freq': ch_freq, 'ch_time': ch_time, 'pn_time': pn_time, 'noise': noise,
                'ref_tx_signal_dmrs': ref_tx_signal_dmrs, 'ref_tx_signal_ptrs': ref_tx_signal_ptrs,
                'ref_rx_signal_dmrs': ref_rx_signal_dmrs, 'ref_rx_signal_ptrs': ref_rx_signal_ptrs}
        return data

    def _generate_noise(self):
        noise_pow = np.power(10, -self._snr_db/10)
        noise = self._rng.normal(loc=0, scale=np.sqrt(noise_pow/2), size=(self._batch_size, self._n_symb, self._n_subc, 2))
        noise = noise[..., 0] + 1j * noise[..., 1]
        return noise

    def _generate_tx_signal(self):
        tx_signal = np.zeros((self._batch_size, self._n_symb, self._n_subc, 2))
        ref_num = int(np.sum(self._ref_mask))
        data_num = self._n_symb * self._n_subc - ref_num
        data_mod_order = int(np.sqrt(self._mod_order))
        data_mod_power_correction = np.sqrt(6/((data_mod_order-1)*(data_mod_order+1)))
        data_mod_index = self._rng.integers(low=0, high=data_mod_order, size=(self._batch_size, data_num, 2))
        data_mod = (data_mod_index - (data_mod_order - 1)/2) * data_mod_power_correction
        tx_signal[:, self._ref_mask, :] = self._ref_signal
        tx_signal[:, np.logical_not(self._ref_mask), :] = data_mod
        tx_signal = tx_signal[..., 0] + 1j * tx_signal[..., 1]
        return tx_signal


def get_dataset_and_dataloader(data_file_name, batch_size, snr_db, mod_order=4, ref_conf_dict=None,
                               num_guard_subcarriers=1024, rnd_seed=0, num_workers=0, is_phase_noise=True,
                               is_channel=True, is_noise=True):
    dataset = ChannelDataset(data_file_name, batch_size, snr_db, mod_order, ref_conf_dict, num_guard_subcarriers,
                             rnd_seed, is_phase_noise, is_channel, is_noise)
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    return dataset, dataloader


if __name__ == "__main__":
    ref_conf_dict = {'dmrs': (0, 3072, 1), 'ptrs': (6, 3072, 48)}
    # dataset = ChannelDataset('channel_mmWave.mat', batch_size=10, snr_db=30, mod_order=16,
    #                          ref_conf_dict=ref_conf_dict, num_guard_subcarriers=1024, rnd_seed=0)
    # it = iter(dataset)
    # a = next(it)
    dataset, dataloader = get_dataset_and_dataloader(data_file_name='channel_mmWave.mat', batch_size=10, snr_db=30,
                                                     mod_order=16, ref_conf_dict=ref_conf_dict, num_guard_subcarriers=1024,
                                                     rnd_seed=0, num_workers=0, is_phase_noise=True, is_channel=True, is_noise=True)
    for it, data in enumerate(dataloader):
        print(it)

