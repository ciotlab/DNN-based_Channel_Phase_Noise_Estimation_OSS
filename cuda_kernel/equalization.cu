// rx_signal (14, 3072, 2), ch_est (3072, 2), pn_est (14,), num_data = 14*3072
extern "C" __global__ void equalization(float *rx_signal, float *ch_est, float *pn_est, int n_subc, int num_data)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_data)
    {
        // Rx signal
        float rx_signal_real = rx_signal[idx * 2];
        float rx_signal_imag = rx_signal[idx * 2 + 1];

        // Channel compensation
        int ch_idx = idx % n_subc;
        float ch_real = ch_est[ch_idx * 2];
        float ch_imag = ch_est[ch_idx * 2 + 1];
        float ch_abs2 = ch_real * ch_real + ch_imag * ch_imag;
        if (ch_abs2 == 0.0f)
            ch_abs2 = 1e-30f;  // Avoid dividing by zero
        float rx_ch_real = ch_real * rx_signal_real + ch_imag * rx_signal_imag; // Multiply conjugated channel
        float rx_ch_imag = - ch_imag * rx_signal_real + ch_real * rx_signal_imag;
        rx_ch_real = rx_ch_real / ch_abs2;
        rx_ch_imag = rx_ch_imag / ch_abs2;

        // Phase noise compensation
        int pn_idx = idx / n_subc;
        float pn_real = cosf(pn_est[pn_idx]);  // real phase noise
        float pn_imag = sinf(pn_est[pn_idx]);  // imag phase noise
        rx_signal[idx * 2] = pn_real * rx_ch_real + pn_imag * rx_ch_imag; // Multiply conjugated phase noise
        rx_signal[idx * 2 + 1] = - pn_imag * rx_ch_real + pn_real * rx_ch_imag;
    }
}