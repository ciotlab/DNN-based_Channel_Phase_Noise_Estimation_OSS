// rx_signal (14, 3072, 2), llr (39104, 6), data_pos (39104,), mcs, llr_num_data (39104)
extern "C" __global__ void llr(float *rx_signal, signed char *llr, int *data_map, int mcs, int num_data)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_data)
    {
        float SCALE_LLR = 62.0f;
        float llr0 = 0.0f;
        float llr1 = 0.0f;
        float llr2 = 0.0f;
        float llr3 = 0.0f;
        float llr4 = 0.0f;
        float llr5 = 0.0f;

        // Equalized Rx signal
        int data_idx = data_map[idx];
        float real = rx_signal[data_idx * 2];
        float imag = rx_signal[data_idx * 2 + 1];

        // LLR computation
        // QPSK, 16QAM, 64QAM
        llr0 = fminf(fmaxf(real, -0.5f), 0.5f) * SCALE_LLR;
        llr1 = fminf(fmaxf(imag, -0.5f), 0.5f) * SCALE_LLR;
        // 16QAM, 64QAM
        if (mcs == 2 || mcs == 3 || mcs == 4 || mcs == 5)
        {
            float llr2_temp = (2.0f / sqrtf(10.0f)) - fabsf(real);
            llr2 = fminf(fmaxf(llr2_temp, -0.5f), 0.5f) * SCALE_LLR;
            float llr3_temp = (2.0f / sqrtf(10.0f)) - fabsf(imag);
            llr3 = fminf(fmaxf(llr3_temp, -0.5f), 0.5f) * SCALE_LLR;
        }
        // 64QAM
        if (mcs == 4 || mcs == 5)
        {
            float llr4_temp1 = (4.0f / sqrtf(42.0f)) - fabsf(real);
            float llr4_temp2 = (2.0f / sqrtf(42.0f)) - fabsf(llr4_temp1);
            llr4 = fminf(fmaxf(llr4_temp2, -0.5f), 0.5f) * SCALE_LLR;
            float llr5_temp1 = (4.0f / sqrtf(42.0f)) - fabsf(imag);
            float llr5_temp2 = (2.0f / sqrtf(42.0f)) - fabsf(llr5_temp1);
            llr5 = fminf(fmaxf(llr5_temp2, -0.5f), 0.5f) * SCALE_LLR;
        }

        int llr_idx = 6 * idx;
        llr[llr_idx] = (signed char)floorf(llr0);
        llr[llr_idx + 1] = (signed char)floorf(llr1);
        llr[llr_idx + 2] = (signed char)floorf(llr2);
        llr[llr_idx + 3] = (signed char)floorf(llr3);
        llr[llr_idx + 4] = (signed char)floorf(llr4);
        llr[llr_idx + 5] = (signed char)floorf(llr5);
    }
}


