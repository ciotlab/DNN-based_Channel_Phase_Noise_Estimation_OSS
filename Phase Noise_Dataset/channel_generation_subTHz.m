seed = 50;
rng(seed)
params.fft_size = 4096; % Number of subcarriers
params.pn_model = 'A'; % Phase noise model
params.carrier_freq = 140E9; % Carrier frequency (Hz)
params.sample_rate = 3932.16e6; % Sample rate (Hz)
params.cp_len = 73.2; % cyclic prefix length (ns), numerlogy 6 regular CP
params.pdp_file = 'PDP_subTHz.mat';
num_samples = 100000;

% Channel generation
ch = NYUChannelGenerator(params);
ch_time = ch.get(num_samples);
freq_conv_mat = fft_matrix(params.fft_size)*sqrt(params.fft_size);
ch_freq = freq_conv_mat * ch_time;

% Phase noise generation
pn = PhaseNoiseGenerator(params);
pn_time = pn.get(num_samples);

save('channel_subTHz.mat', 'ch_freq', 'pn_time', '-v7.3');
