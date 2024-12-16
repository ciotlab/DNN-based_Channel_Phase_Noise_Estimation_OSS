import os
import torch
import numpy as np
import tensorrt as trt
from pathlib import Path
import cupy as cp
import cupyx as cpx
from dataloader import SampleDataLoader


class Processor(object):
    def __init__(self, ch_engine, pn_engine, device_id=0, ret_eq_signal=True, ret_evm=True, n_symb=14, n_subc=3072, batch_size=1, eq_block_size=256, llr_block_size=256):
        self.batch_size = batch_size
        self.device_id = device_id
        self.device = cp.cuda.Device(self.device_id)
        with self.device:
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.ret_eq_signal = ret_eq_signal
        self.ret_evm = ret_evm
        # Slot structure
        self.n_symb = n_symb  # number of symbol
        self.n_subc = n_subc  # number of subcarrier
        self.slot_shape = (self.n_symb, self.n_subc)
        self.dmrs_pos = np.full(self.slot_shape, fill_value=False)
        self.dmrs_pos[0, :] = True
        self.dmrs_map = self.dmrs_pos.flatten().nonzero()[0]
        self.n_dmrs = self.dmrs_map.size
        self.ptrs_pos = np.full(self.slot_shape, fill_value=False)
        self.ptrs_pos[:, 6:3072:48] = True
        self.ptrs_map = self.ptrs_pos.flatten().nonzero()[0]
        self.n_ptrs = self.ptrs_map.size
        self.data_pos = np.logical_not(np.logical_or(self.dmrs_pos, self.ptrs_pos))
        self.data_map = self.data_pos.flatten().nonzero()[0]
        self.n_data = self.data_map.size
        # Channel post processing parameter
        self.fft_size = 4096  # FFT size
        self.num_guard_subc = 1024  # Number of guard subcarriers
        self.ch_est_len = 256  # Estimation length of time-domain channel
        # Allocate memory
        self.rx_signal_cpu = cpx.empty_pinned(shape=(*self.slot_shape, 2), dtype=cp.float32)  # Pinned host memory for rx signal (symbol, subcarrier, re/im)
        self.rx_signal_gpu = cp.empty(shape=(*self.slot_shape, 2), dtype=cp.float32)  # GPU memory for rx signal (symbol, subcarrier, re/im)
        self.ch_input_gpu = cp.empty(shape=(2 * self.n_subc,), dtype=cp.float32)  # GPU memory for input to channel estimator (re/im * num_dmrs)
        self.pn_input_gpu = cp.empty(shape=(2 * self.n_ptrs,), dtype=cp.float32)  # GPU memory for input to phase noise estimator (re/im * num_ptrs)
        self.ch_output_gpu = cp.empty(shape=(2 * self.ch_est_len,), dtype=cp.float32)  # GPU memory for output from channel estimator (re/im * num_time_domain_sample)
        self.pn_output_gpu = cp.empty(shape=(self.n_symb,), dtype=cp.float32)  # GPU memory for output from phase noise estimator (re/im * num_symbol)
        self.ch_est_gpu = cp.empty(shape=(self.n_subc, 2), dtype=cp.float32)  # GPU memory for post-processed estimated channel (subcarrier, re/im)
        self.llr_cpu = cpx.empty_pinned(shape=(self.n_data, 6), dtype=cp.int8)  # Pinned host memory for LLR (data symbol, bit)
        self.llr_gpu = cp.empty(shape=(self.n_data, 6), dtype=cp.int8)  # GPU memory for LLR (data symbol, bit)
        self.data_map_gpu = cp.asarray(self.data_map, dtype=cp.int32)
        # Setup TensorRT engine
        self.ch_trt_context = ch_engine.create_execution_context()
        self.pn_trt_context = pn_engine.create_execution_context()
        # Setup equalization CUDA kernel
        cuda_kernel_path = Path(__file__).parents[0].resolve() / 'cuda_kernel'
        with open(cuda_kernel_path / 'equalization.cu', 'r') as f:
            kernel_code = f.read()
        module = cp.RawModule(code=kernel_code)
        self.eq_kernel = module.get_function('equalization')
        self.eq_num_data = self.n_symb * self.n_subc
        self.eq_block_size = eq_block_size
        self.eq_grid_size = (self.eq_num_data + self.eq_block_size - 1) // self.eq_block_size
        # Setup LLR CUDA kernel
        with open(cuda_kernel_path / 'llr.cu', 'r') as f:
            kernel_code = f.read()
        module = cp.RawModule(code=kernel_code)
        self.llr_kernel = module.get_function('llr')
        self.llr_num_data = self.n_data
        self.llr_block_size = llr_block_size
        self.llr_grid_size = (self.llr_num_data + self.llr_block_size - 1) // self.llr_block_size

    def run(self, data):
        with self.device, self.stream:
            rx_signal, control_signal, info = self.parse_data(data)  # Parsing data
            np.copyto(dst=self.rx_signal_cpu, src=rx_signal)  # Copy the rx signal to the pinned memory
            self.rx_signal_gpu.set(self.rx_signal_cpu, stream=self.stream)  # Send rx signal to GPU
            ch_std = self.ch_pn_estimation()  # Channel and phase noise estimation
            self.ch_postprocessing(ch_std)  # Postprocessing channel
            self.equalization()  # Perform equalization
            self.llr_calculation(info['mcs'])  # Compute LLR
            self.llr_cpu = self.llr_gpu.get(stream=self.stream)
            ret = {'control': control_signal, 'llr': self.llr_cpu}
            if self.ret_eq_signal:
                ret['ret_eq_signal'] = self.get_eq_signal()
            if self.ret_evm:
                ret['evm'] = self.evm_calculation()
        return ret

    def parse_data(self, data):
        control_signal, rx_signal = np.split(data, indices_or_sections=[16], axis=0)
        mcs = control_signal[9]
        frame_num = control_signal[10] // (2 ** 16)
        subframe_num = control_signal[10] % (2 ** 16)
        slot_num = control_signal[11] // (2 ** 16)
        info = {'mcs': mcs, 'frame_num': frame_num, 'subframe_num': subframe_num, 'slot_num': slot_num}
        rx_signal = rx_signal.astype(np.float32) / (2.0 ** 11)
        rx_signal = np.reshape(rx_signal, newshape=(*self.slot_shape, 2))
        return rx_signal, control_signal, info

    def ch_pn_estimation(self):
        self.ch_input_gpu = self.rx_signal_gpu[self.dmrs_pos, :].transpose().flatten()  # (re/im, n_dmrs) (2, 3072)
        self.pn_input_gpu = self.rx_signal_gpu[self.ptrs_pos, :].transpose().flatten()  # (re/im, n_ptrs) (2, 896)
        ch_std = cp.sqrt(cp.sum(self.ch_input_gpu ** 2.0) / self.n_subc)
        self.ch_input_gpu = self.ch_input_gpu / ch_std
        self.pn_input_gpu = self.pn_input_gpu / ch_std
        # Channel estimation
        self.ch_trt_context.set_tensor_address('input', self.ch_input_gpu.data.ptr)
        self.ch_trt_context.set_tensor_address('output', self.ch_output_gpu.data.ptr)
        self.ch_trt_context.execute_async_v3(stream_handle=self.stream.ptr)
        # Phase noise estimation
        self.pn_trt_context.set_tensor_address('input', self.pn_input_gpu.data.ptr)
        self.pn_trt_context.set_tensor_address('output', self.pn_output_gpu.data.ptr)
        self.pn_trt_context.execute_async_v3(stream_handle=self.stream.ptr)
        return ch_std

    def ch_postprocessing(self, ch_std):
        ch_est = self.ch_output_gpu.reshape((2, self.ch_est_len))  # re/im, sample
        ch_est = ch_est[0, :] + 1j * ch_est[1, :]
        ch_est = ch_est * ch_std
        ch_est = cp.concatenate((ch_est, cp.zeros(shape=(self.fft_size - self.ch_est_len,), dtype=np.complex64)), axis=0)
        ch_est = cp.fft.fftshift(cp.fft.fft(ch_est, norm='ortho'), axes=-1)
        ch_est = ch_est[self.num_guard_subc // 2: self.fft_size - self.num_guard_subc // 2]
        self.ch_est_gpu[:, 0], self.ch_est_gpu[:, 1] = cp.real(ch_est), cp.imag(ch_est)

    def equalization(self):
        arg = (self.rx_signal_gpu, self.ch_est_gpu, self.pn_output_gpu, self.n_subc, self.eq_num_data)  # rx_signal_gpu (14, 3072, 2), ch_est_gpu (3072, 2), pn_output_gpu (14,)
        self.eq_kernel(grid=(self.eq_grid_size,), block=(self.eq_block_size,), args=arg, stream=self.stream)

    def llr_calculation(self, mcs):
        arg = (self.rx_signal_gpu, self.llr_gpu, self.data_map_gpu, mcs, self.llr_num_data)  # rx_signal_gpu (14, 3072, 2), llr_gpu (39104, 6), data_map_gpu (39104,), mcs, llr_num_data (39104)
        self.llr_kernel(grid=(self.llr_grid_size,), block=(self.llr_block_size,), args=arg, stream=self.stream)

    def get_eq_signal(self):
        eq_signal = self.rx_signal_gpu[self.data_pos, :]
        eq_signal = eq_signal.get(stream=self.stream)
        return eq_signal

    def evm_calculation(self):
        ptrs = self.rx_signal_gpu[self.ptrs_pos, :]
        ptrs = ptrs[:, 0] + 1j * ptrs[:, 1]
        noise = ptrs - 1.0
        noise_power = cp.mean(cp.clip(cp.square(cp.abs(noise)), a_min=None, a_max=16.0))
        signal_power = 1.0 + noise_power
        sinr = signal_power / noise_power
        MAX_SINR_LINEAR = 1e7
        sinr = cp.clip(sinr, a_min=None, a_max=MAX_SINR_LINEAR)
        evm = -10.0 * cp.log10(sinr)
        evm = evm.get(stream=self.stream).astype(np.float32)
        return evm

    def equalization_cupy(self):
        pn_est = cp.exp(1j * self.pn_output_gpu)
        ch_est = self.ch_est_gpu[:, 0] + 1j * self.ch_est_gpu[:, 1]
        rx_signal = self.rx_signal_gpu[:, :, 0] + 1j * self.rx_signal_gpu[:, :, 1]
        rx_signal = rx_signal / ch_est[cp.newaxis, :]
        rx_signal = rx_signal * cp.conj(pn_est[:, cp.newaxis])
        self.rx_signal_gpu[:, :, 0], self.rx_signal_gpu[:, :, 1] = cp.real(rx_signal), cp.imag(rx_signal)

    def llr_calculation_cupy(self, mcs):
        SCALE_LLR = 62.0
        x = self.rx_signal_gpu[self.data_pos, :]
        xr, xi = x[:, 0], x[:, 1]
        # Initialize LLR arrays
        llr = cp.zeros((self.n_data, 6))
        # LLR computation
        # QPSK, 16QAM, 64QAM
        llr[:, 0] = cp.clip(xr, a_min=-0.5, a_max=0.5) * SCALE_LLR
        llr[:, 1] = cp.clip(xi, a_min=-0.5, a_max=0.5) * SCALE_LLR
        # 16QAM, 64QAM
        if mcs in [2, 3, 4, 5]:
            llr[:, 2] = cp.clip((2 / np.sqrt(10.0)) - cp.abs(xr), a_min=-0.5, a_max=0.5) * SCALE_LLR
            llr[:, 3] = cp.clip((2 / np.sqrt(10.0)) - cp.abs(xi), a_min=-0.5, a_max=0.5) * SCALE_LLR
        # 64QAM
        if mcs in [4, 5]:
            llr4_temp1 = (4 / cp.sqrt(42.0)) - cp.abs(xr)
            llr4_temp2 = (2 / cp.sqrt(42.0)) - cp.abs(llr4_temp1)
            llr[:, 4] = cp.clip(llr4_temp2, a_min=-0.5, a_max=0.5) * SCALE_LLR
            llr5_temp1 = (4 / cp.sqrt(42.0)) - cp.abs(xi)
            llr5_temp2 = (2 / cp.sqrt(42.0)) - cp.abs(llr5_temp1)
            llr[:, 5] = cp.clip(llr5_temp2, a_min=-0.5, a_max=0.5) * SCALE_LLR
        self.llr_gpu = cp.floor(llr).astype(cp.int8)

    @torch.no_grad()
    def run_torch(self, data):  # full pytorch implementation (slow)
        rx_signal, control_signal, info = self.parse_data(data)
        # prepare input to the neural networks
        ch_in = torch.tensor(rx_signal[0, :, :].transpose()[np.newaxis, :, :]).cuda()  # batch, re/im, subcarrier (1, 2, 3072)
        pn_in = torch.tensor(rx_signal[:, 6:3072:48, :].reshape((-1, 2)).transpose()[np.newaxis, :, :]).cuda()  # batch, re/im, (symbol, subcarrier) (1, 2, 896)
        ch_pwr_avg = torch.sum(torch.square(ch_in), dim=(1, 2), keepdim=True) / ch_in.shape[2]
        ch_in /= torch.sqrt(ch_pwr_avg)
        pn_in /= torch.sqrt(ch_pwr_avg)
        # load torch model
        path = Path(__file__).parents[0].resolve() / 'torch_model'
        ch_model = torch.load(path / 'transformer_ch.pt').cuda().eval()
        pn_model = torch.load(path / 'transformer_pn.pt').cuda().eval()
        # inference
        ch_est = ch_model(ch_in)
        ch_est = torch.complex(ch_est[0, 0, :], ch_est[0, 1, :])
        ch_est = ch_est * torch.sqrt(ch_pwr_avg[0, 0])
        ch_est = torch.cat(tensors=(ch_est, torch.zeros(4096 - ch_est.shape[0]).cuda()), dim=0)
        ch_est = torch.fft.fftshift(torch.fft.fft(ch_est, dim=-1, norm='ortho'), dim=(-1,))
        ch_est = ch_est[512: 4096 - 512]
        pn_est = pn_model(pn_in)
        pn_est = torch.exp(1j * pn_est[0, 0, :])
        # equalization
        rx_signal = torch.tensor(rx_signal).cuda()
        rx_signal = torch.complex(rx_signal[:, :, 0], rx_signal[:, :, 1])  # (14, 3072)
        rx_signal = rx_signal / ch_est[None, :]
        equalized_rx_signal = rx_signal * torch.conj(pn_est[:, None])
        # LLR
        llr = self.llr_calculation_torch(rx_signal, info['mcs'])
        # EVM
        evm = self.evm_calculation_torch(rx_signal)
        # Extract data position
        data_pos = np.ones((14, 3072)).astype(np.bool_)
        data_pos[:, 6:3072:48] = False
        data_pos[0, :] = False
        equalized_rx_signal = equalized_rx_signal[data_pos].cpu().numpy().reshape((13, 3008))
        equalized_rx_signal = np.stack((np.real(equalized_rx_signal), np.imag(equalized_rx_signal)), axis=1)
        llr = llr[data_pos, :].reshape((13, 3008, 6))
        return control_signal, llr, equalized_rx_signal, evm

    def llr_calculation_torch(self, x, mcs):
        SCALE_LLR = 62.0
        xr, xi = torch.real(x), torch.imag(x)
        # Initialize LLR arrays
        llr = torch.zeros((6, *x.shape)).cuda()
        # LLR computation
        # QPSK, 16QAM, 64QAM
        llr[0] = torch.clip(xr, -0.5, 0.5) * SCALE_LLR
        llr[1] = torch.clip(xi, -0.5, 0.5) * SCALE_LLR
        # 16QAM, 64QAM
        if mcs in [2, 3, 4, 5]:
            llr[2] = torch.clip((2 / np.sqrt(10.0)) - torch.abs(xr), -0.5, 0.5) * SCALE_LLR
            llr[3] = torch.clip((2 / np.sqrt(10.0)) - torch.abs(xi), -0.5, 0.5) * SCALE_LLR
        # 64QAM
        if mcs in [4, 5]:
            llr4_temp1 = (4 / np.sqrt(42.0)) - torch.abs(xr)
            llr4_temp2 = (2 / np.sqrt(42.0)) - torch.abs(llr4_temp1)
            llr[4] = torch.clip(llr4_temp2, -0.5, 0.5) * SCALE_LLR
            llr5_temp1 = (4 / np.sqrt(42.0)) - torch.abs(xi)
            llr5_temp2 = (2 / np.sqrt(42.0)) - torch.abs(llr5_temp1)
            llr[5] = torch.clip(llr5_temp2, -0.5, 0.5) * SCALE_LLR
        llr = torch.floor(llr).to(torch.int8)
        llr = llr.cpu().numpy()
        llr = np.transpose(llr, axes=(1, 2, 0))
        return llr

    def evm_calculation_torch(self, x):
        ptrs = x[:, 6:3072:48]
        noise = ptrs - 1.0
        noise_power = torch.mean(torch.clip(torch.square(torch.abs(noise)), min=None, max=16.0))
        signal_power = 1.0 + noise_power
        sinr = signal_power / noise_power
        MAX_SINR_LINEAR = 1e7
        sinr = torch.clip(sinr, min=None, max=MAX_SINR_LINEAR)
        evm = -10.0 * torch.log10(sinr)
        evm = evm.cpu().numpy().astype(np.float32)
        return evm


def load_engine(file_name):
    with trt.Logger(trt.Logger.ERROR) as logger, trt.Runtime(logger) as runtime:
        engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
        with open(engine_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
    return engine


def test_processor(device_id=0, ret_eq_signal=False, ret_evm=False):
    ch_engine = load_engine('transformer_ch')
    pn_engine = load_engine('transformer_pn')
    p = Processor(ch_engine, pn_engine, device_id=device_id, ret_eq_signal=ret_eq_signal, ret_evm=ret_evm)
    data_loader = SampleDataLoader('sample_no_noise')
    for data in data_loader:
        res = p.run(data)


if __name__ == "__main__":
    os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda-12.2/bin/"  # Add nvcc path
    test_processor(device_id=0, ret_eq_signal=True, ret_evm=True)

