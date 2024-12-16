seed = 50;
rng(seed)
params.fft_size = 2048; % Number of subcarriers
params.pn_model = 'A'; % Phase noise model
params.carrier_freq = 140e9; % Carrier frequency
params.sample_rate = 3932.16e6; % Sample rate
%params.delay_profile = 'TDL-C'; % TDL delay profile
%params.delay_spread = 300e-9; % TDL delay spread (sec)
params.data_mod_order = 64; % Data modulation order
params.snr_db = inf; % SNR (dB)
%params.tdl_seed_base = seed;
params.ref_alloc = ... % Reference signal configuration 1 (start, step, end)
    [1, 1,  128;
     3, 6, 2048];
% params.ref_alloc = ... % Reference signal configuration 2 (start, step, end)
%     [1, 12, 2048
%     1, 13, 2048];

% Channel generation
num_signal = 1;
ofdm = OFDMSystem(params);
[ref_sym, ref_loc, ref_mat, data_loc] = ofdm.get_structure();
[ofdm, rx_time, rx_freq, pn_time, ch_freq] = ofdm.receive(num_signal);

% Estimation
num_ch_coef = 128; % Number of channel coefficients 
num_pn_coef = 64; % Number of phase noise coeffcients

% % LS channel-only algorithm
% ch = LSChOnlyAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym);
% [mu_ch, u_ch, h_ch, nu_ch, v_ch, p_ch] = ch.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_ch, h_ch);
% %ofdm.show_phase_noise_and_channel(1);
% %ofdm.show_constellation(1);
% [ph_nmse_ch, ch_nmse_ch, evm_ch, ber_ch] = ofdm.cal_performance_measure();

% LS algorithm
ls_num_iter = 4;
ls = LSAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ls_num_iter);
[mu_ls, u_ls, h_ls, nu_ls, v_ls, p_ls] = ls.estimate(rx_freq);
ofdm = ofdm.set_estimated_phase_noise_and_channel(v_ls, h_ls);
[pn, est_pn_ls, ch, est_ch_ls] = ofdm.show_phase_noise_and_channel(1);
ofdm.show_constellation(1);
[ph_nmse_ls, ch_nmse_ls, evm_ls, ber_ls] = ofdm.cal_performance_measure();
  
% % MM algorithm
% mm_num_iter = 64;
% mm = MMAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, mm_num_iter);
% [mu_mm, u_mm, h_mm, nu_mm, v_mm, p_mm] = mm.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_mm, h_mm);
% [pn, est_pn_mm, ch, est_ch_mm] = ofdm.show_phase_noise_and_channel(1);
% % ofdm.show_constellation(1);
% [ph_nmse_mm, ch_nmse_mm, evm_mm, ber_mm] = ofdm.cal_performance_measure();
% 
% % GLS algorithm
% R_nu = 0.1*eye(num_pn_coef) + ones(num_pn_coef);
% sigma_nh_sq = 0.03;
% rho = 1;
% gls_num_iter = 1;
% gls = GLSAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ...
%     R_nu, sigma_nh_sq, rho, gls_num_iter);
% [mu_gls, u_gls, h_gls, nu_gls, v_gls, p_gls] = gls.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_gls, h_gls);
% [pn, est_pn_gls, ch, est_ch_gls] = ofdm.show_phase_noise_and_channel(1);
% % ofdm.show_constellation(1);
% [ph_nmse_gls, ch_nmse_gls, evm_gls, ber_gls] = ofdm.cal_performance_measure();
% 
% %LMMSE algorithm
% R_nu = 0.1*eye(num_pn_coef) + ones(num_pn_coef);
% sigma_nh_sq = 0.03;
% rho = 1;
% lmmse_num_iter = 1;
% lmmse = LMMSEAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ...
%     R_nu, sigma_nh_sq, rho, lmmse_num_iter);
% [mu_lmmse, u_lmmse, h_lmmse, nu_lmmse, v_lmmse, p_lmmse] = lmmse.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_lmmse, h_lmmse);
% [pn, est_pn_lmmse, ch, est_ch_lmmse] = ofdm.show_phase_noise_and_channel(1);
% %ofdm.show_constellation(1);
% [ph_nmse_lmmse, ch_nmse_lmmse, evm_lmmse, ber_lmmse] = ofdm.cal_performance_measure();