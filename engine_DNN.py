from dataset import get_dataset_and_dataloader
import torch
import torch.nn.functional as F
import torch_tensorrt
import math
import numpy as np
from pathlib import Path
import yaml
from einops import repeat, rearrange
from model.transformer import Transformer, ConditionNetwork
from utils.plot_signal import plot_signal
from timeit import default_timer as timer
import wandb


class Engine:
    def __init__(self, conf_file, device='cuda:0', use_wandb=True, wandb_proj='DNN_channel_estimation'):
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        with open(conf_path) as f:
            self._conf = yaml.safe_load(f)
        self._device = device
        self._use_wandb = use_wandb
        self._wandb_proj = wandb_proj
        if self._use_wandb:
            wandb.init(project=self._wandb_proj, config=self._conf)
            self._conf = wandb.config
        # Get dataset and dataloader
        self._dataset, self._dataloader = get_dataset_and_dataloader(**self._conf['dataset'])
        # Channel estimation network
        ch_cond_netw = ConditionNetwork(length=self._dataset.n_dmrs, **self._conf['ch_estimation']['cond'])
        self._ch_tf = Transformer(**self._conf['ch_estimation']['transformer'], cond_net=ch_cond_netw).to(self._device)
        # Phase noise estimation network
        pn_cond_netw = ConditionNetwork(length=self._dataset.n_ptrs, **self._conf['pn_estimation']['cond'])  # n_ptrs: 896 (64 per symbol)
        self._pn_tf = Transformer(**self._conf['pn_estimation']['transformer'], cond_net=pn_cond_netw).to(self._device)
        # Optimizer
        lr = self._conf['training']['lr']
        weight_decay = self._conf['training']['weight_decay']
        self._max_norm = self._conf['training']['max_norm']
        self._num_iter = self._conf['training']['num_iter']
        ch_tf_params = [p for n, p in self._ch_tf.named_parameters() if p.requires_grad]
        pn_tf_params = [p for n, p in self._pn_tf.named_parameters() if p.requires_grad]
        self._ch_optimizer = torch.optim.AdamW([{"params": ch_tf_params}], lr=lr, weight_decay=weight_decay)
        self._pn_optimizer = torch.optim.AdamW([{"params": pn_tf_params}], lr=lr, weight_decay=weight_decay)
        # Tensor-RT
        self._ch_tf_trt = None
        self._pn_tf_trt = None
        self._trt_ch_input_shape = None
        self._trt_pn_input_shape = None

    def train(self):
        for it, data in enumerate(self._dataloader):
            self._ch_tf.train()
            self._pn_tf.train()
            ch_in, pn_in, ch_pwr_avg, true_ch, true_ch_reim, true_pn = self._preprocess_data(data)
            # Channel update
            ch_est = self._ch_tf(ch_in)
            ch_loss = torch.mean(torch.square(true_ch_reim - ch_est))
            self._ch_optimizer.zero_grad()
            ch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._ch_tf.parameters(), max_norm=self._max_norm)
            self._ch_optimizer.step()
            # Phase noise update
            pn_est = self._pn_tf(pn_in)[:, 0, :]
            pn_loss = torch.mean(torch.sum(torch.square(true_pn - pn_est), dim=1))
            self._pn_optimizer.zero_grad()
            pn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._pn_tf.parameters(), max_norm=self._max_norm)
            self._pn_optimizer.step()
            if (it + 1) % self._conf['training']['logging_step'] == 0:
                self._logging(it, ch_loss, ch_pwr_avg, ch_est, true_ch, pn_loss, pn_est, true_pn)
            if it >= self._num_iter - 1:
                break

    def _preprocess_data(self, data):
        # Channel data
        ch_est_length = self._conf['ch_estimation']['transformer']['length']
        ch_in = torch.tensor(data['ref_rx_signal_dmrs'] * np.conjugate(data['ref_tx_signal_dmrs']),
                             dtype=torch.cfloat).to(self._device)
        ch_pwr_avg = torch.mean(torch.square(torch.abs(ch_in)), dim=1, keepdim=True)
        ch_in = ch_in / torch.sqrt(ch_pwr_avg)
        ch_in = torch.stack((torch.real(ch_in), torch.imag(ch_in)), dim=1)  # batch, re/im, data
        true_ch = torch.tensor(data['ch_time'], dtype=torch.cfloat).to(self._device)
        true_ch_reim = true_ch[:, :ch_est_length] / torch.sqrt(ch_pwr_avg)
        true_ch_reim = torch.stack((torch.real(true_ch_reim), torch.imag(true_ch_reim)), dim=1)  # batch, re/im, data
        # Phase noise data
        pn_in = torch.tensor(data['ref_rx_signal_ptrs'] * np.conjugate(data['ref_tx_signal_ptrs']),
                             dtype=torch.cfloat).to(self._device)
        pn_in = torch.stack((torch.real(pn_in), torch.imag(pn_in)), dim=1)  # batch, re/im, data
        true_pn = torch.tensor(data['pn_time'], dtype=torch.cfloat).to(self._device)
        true_pn = true_pn * torch.conj(true_pn[:, :1])
        true_pn = torch.angle(true_pn)  # batch, data
        return ch_in, pn_in, ch_pwr_avg, true_ch, true_ch_reim, true_pn

    @torch.no_grad()
    def _logging(self, it, ch_loss, ch_pwr_avg, ch_est, true_ch, pn_loss, pn_est, true_pn):
        # Channel NMSE
        ch_est = torch.complex(ch_est[:, 0, :], ch_est[:, 1, :])
        ch_est = ch_est * torch.sqrt(ch_pwr_avg)
        ch_len = true_ch.shape[1]
        ch_est_len = ch_est.shape[1]
        ch_est = torch.cat((ch_est, torch.zeros((ch_est.shape[0], ch_len - ch_est_len)).to(self._device)), dim=-1)
        ch_mse = torch.mean(torch.sum(torch.square(torch.absolute(true_ch - ch_est)), dim=1))
        ch_var = torch.mean(torch.sum(torch.square(torch.absolute(true_ch)), dim=1))
        ch_nmse = ch_mse / ch_var
        # Phase noise NMSE
        pn_mse = torch.mean(torch.sum(torch.square(true_pn - pn_est), dim=1))
        pn_var = torch.mean(torch.sum(torch.square(true_pn), dim=1))
        pn_nmse = pn_mse / pn_var
        log = {'ch_loss': ch_loss, 'pn_loss': pn_loss, 'ch_nmse': ch_nmse, 'pn_nmse': pn_nmse}
        print(f"iteration:{it + 1}, ch_loss:{log['ch_loss']}, pn_loss:{log['pn_loss']}, ch_nmse:{log['ch_nmse']}, pn_nmse:{log['pn_nmse']}")
        if self._use_wandb:
            wandb.log(log)
        if (it + 1) % self._conf['training']['evaluation_step'] == 0:
            show_batch_size = self._conf['training']['evaluation_batch_size']
            ch = true_ch[:show_batch_size].detach().cpu().numpy()
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy()
            ch_freq = np.fft.fftshift(np.fft.fft(ch, axis=-1, norm='ortho'), axes=(-1,))
            ch_est_freq = np.fft.fftshift(np.fft.fft(ch_est, axis=-1, norm='ortho'), axes=(-1,))
            pn = true_pn[:show_batch_size].detach().cpu().numpy()
            pn_est = pn_est[:show_batch_size].detach().cpu().numpy()
            sig_dict = {}
            sig_dict['ch_time_est_real'] = {'data': ch_est, 'type': 'real', 'x_range': (0, ch_est_len)}
            sig_dict['ch_time_real'] = {'data': ch, 'type': 'real', 'x_range': (0, ch_est_len)}
            sig_dict['ch_time_est_imag'] = {'data': ch_est, 'type': 'imag', 'x_range': (0, ch_est_len)}
            sig_dict['ch_time_imag'] = {'data': ch, 'type': 'imag', 'x_range': (0, ch_est_len)}
            sig_dict['ch_freq_est_magnitude'] = {'data': ch_est_freq, 'type': 'magnitude'}
            sig_dict['ch_freq_magnitude'] = {'data': ch_freq, 'type': 'magnitude'}
            sig_dict['ch_freq_est_phase'] = {'data': ch_est_freq, 'type': 'phase'}
            sig_dict['ch_freq_phase'] = {'data': ch_freq, 'type': 'phase'}
            sig_dict['pn_est'] = {'data': pn_est, 'type': 'scalar'}
            sig_dict['pn'] = {'data': pn, 'type': 'scalar'}
            f = plot_signal(sig_dict, shape=(5, 2))
            f.show()
            if self._use_wandb:
                wandb.log({'estimation': wandb.Image(f)})
            self.save_model('transformer')

    def save_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self._ch_tf, path / (file_name + '_ch.pt'))
        torch.save(self._pn_tf, path / (file_name + '_pn.pt'))

    def load_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._ch_tf = torch.load(path / (file_name + '_ch.pt'))
        self._pn_tf = torch.load(path / (file_name + '_pn.pt'))
        lr = self._conf['training']['lr']
        weight_decay = self._conf['training']['weight_decay']
        ch_tf_params = [p for n, p in self._ch_tf.named_parameters() if p.requires_grad]
        pn_tf_params = [p for n, p in self._pn_tf.named_parameters() if p.requires_grad]
        self._ch_optimizer = torch.optim.AdamW([{"params": ch_tf_params}], lr=lr, weight_decay=weight_decay)
        self._pn_optimizer = torch.optim.AdamW([{"params": pn_tf_params}], lr=lr, weight_decay=weight_decay)

    def estimate_inference_perf(self, batch_size, num_iter, tensorrt=False, tensorrt_compile=False):
        self._ch_tf.eval()
        self._pn_tf.eval()
        data = next(iter(self._dataloader))
        ch_in, pn_in, ch_pwr_avg, true_ch, true_ch_reim, true_pn = self._preprocess_data(data)
        ch_in = torch.zeros((batch_size, ch_in.shape[1], ch_in.shape[2])).to(self._device)
        pn_in = torch.zeros((batch_size, pn_in.shape[1], pn_in.shape[2])).to(self._device)
        if tensorrt:
            if tensorrt_compile:
                self.tensorrt_compile(batch_size)
            ch_model = self._ch_tf_trt
            pn_model = self._pn_tf_trt
        else:
            ch_model = self._ch_tf
            pn_model = self._pn_tf
        start = timer()
        for it in range(num_iter):
            ch_est = ch_model(ch_in)
            pn_est = pn_model(pn_in)
        end = timer()
        elapsed_time = end - start  # second
        latency_per_batch = elapsed_time / num_iter * 1000.0  # ms
        throughput = (batch_size * num_iter) / elapsed_time  # Hz
        print(f"batch_size: {batch_size}, latency_per_batch: {latency_per_batch:.4f} ms, throughput: {throughput:.4f} Hz")

    def tensorrt_compile(self, batch_size):
        self._ch_tf.eval()
        self._pn_tf.eval()
        data = next(iter(self._dataloader))
        ch_in, pn_in, ch_pwr_avg, true_ch, true_ch_reim, true_pn = self._preprocess_data(data)
        self._trt_ch_input_shape = (batch_size, ch_in.shape[1], ch_in.shape[2])
        self._trt_pn_input_shape = (batch_size, pn_in.shape[1], pn_in.shape[2])
        ch_inputs = torch_tensorrt.Input(shape=self._trt_ch_input_shape, dtype=torch.float32)
        pn_inputs = torch_tensorrt.Input(shape=self._trt_pn_input_shape, dtype=torch.float32)
        self._ch_tf_trt = torch_tensorrt.compile(self._ch_tf, ir="dynamo", inputs=[ch_inputs])
        self._pn_tf_trt = torch_tensorrt.compile(self._pn_tf, ir="dynamo", inputs=[pn_inputs])

    def save_tensorrt_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        ch_inputs = [torch.zeros(self._trt_ch_input_shape, dtype=torch.float32).to(self._device)]
        pn_inputs = [torch.zeros(self._trt_pn_input_shape, dtype=torch.float32).to(self._device)]
        torch_tensorrt.save(self._ch_tf_trt, path / (file_name + '_ch.ep'), inputs=ch_inputs)
        torch_tensorrt.save(self._pn_tf_trt, path / (file_name + '_pn.ep'), inputs=pn_inputs)

    def load_tensorrt_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._ch_tf_trt = torch.export.load(path / (file_name + '_ch.ep')).module()
        self._pn_tf_trt = torch.export.load(path / (file_name + '_pn.ep')).module()


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)

    # Training
    conf_file = 'config.yaml'
    engine = Engine(conf_file, device='cuda:0', use_wandb=True, wandb_proj='DNN_channel_estimation')
    engine.train()

    # # Inference performance test
    # conf_file = 'config.yaml'
    # engine = Engine(conf_file, device='cuda:0', use_wandb=False, wandb_proj='DNN_channel_estimation')
    # engine.estimate_inference_perf(batch_size=4, num_iter=1000, tensorrt=True, tensorrt_compile=True)

    # # Tensor-RT compile
    # conf_file = 'config.yaml'
    # engine = Engine(conf_file, device='cuda:0', use_wandb=False, wandb_proj='DNN_channel_estimation')
    # engine.load_model(file_name='transformer')
    # engine.tensorrt_compile(batch_size=8)
    # engine.save_tensorrt_model(file_name='transformer_trt')
    # #engine.load_tensorrt_model(file_name='transformer_trt')




