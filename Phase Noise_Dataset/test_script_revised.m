params.fft_size = 4096; % Number of subcarriers
params.pn_model = 'A'; % Phase noise model
params.carrier_freq = 140E9; % Carrier frequency (Hz)
params.sample_rate = 3932.16e6; % Sample rate (Hz)
params.cp_len = 73.2; % cyclic prefix length (ns), numerlogy 6 regular CP
params.data_mod_order = 64; % Data modulation order
params.snr_db = 40; % SNR (dB)
%params.ref_type = 'SBC'; % SBC: single block and comb, MB: multi-block
%params.ref_block_len = 256;
%params.ref_spacing = 6;
% params.ref_type = 'MB'; % SBC: single block and comb, MB: multi-block
% params.ref_block_len = 2;
% params.ref_spacing = 12;

num_signal = 200;

% Estimation
num_ch_coef = 256; % Number of channel coefficients 
num_pn_coef = 256; % Number of phase noise coeffcients

result.ref_block_len = [];
result.ref_spacing = [];
result.ref_density = [];
result.ls.pn_nmse = [];
result.ls.ch_nmse = [];
result.ls.ber = [];
result.mm.pn_nmse = [];
result.mm.ch_nmse = [];
result.mm.ber = [];
result.gls.pn_nmse = [];
result.gls.ch_nmse = [];
result.gls.ber = [];
result.lmmse.pn_nmse = [];
result.lmmse.ch_nmse = [];
result.lmmse.ber = [];


%%%%%%%%%%%%%%%%%%% SBC %%%%%%%%%%%%%%%%%%%%%%%
params.ref_type = 'SBC';
ref_spacing_conf = [12, 6, 4];
ref_block_len_conf = [128, 256, 384, 512, 640, 768, 896];

for ref_spacing_idx = 1:numel(ref_spacing_conf)
for ref_block_len_idx = 1:numel(ref_block_len_conf)

fprintf('ref_spacing_idx:%d, ref_block_len_idx:%d\n', ref_spacing_idx, ref_block_len_idx);

params.ref_block_len = ref_block_len_conf(ref_block_len_idx);
params.ref_spacing = ref_spacing_conf(ref_spacing_idx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% MB %%%%%%%%%%%%%%%%%%%%%%%
% params.ref_type = 'MB';
% ref_spacing_conf = [24, 12];
% ref_block_ratio_conf = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12];
% 
% for ref_spacing_idx = 1:numel(ref_spacing_conf)
% for ref_block_ratio_idx = 1:numel(ref_block_ratio_conf)
% 
% fprintf('ref_spacing_idx:%d, ref_block_ratio_idx:%d\n', ref_spacing_idx, ref_block_ratio_idx);
% 
% params.ref_spacing = ref_spacing_conf(ref_spacing_idx);
% params.ref_block_len = floor(params.ref_spacing * ref_block_ratio_conf(ref_block_ratio_idx));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% OFDM processing
random_seed = 0;
ofdm = OFDMSystem(params);
[ref_sym, ref_loc, ref_mat, data_loc, ref_density] = ofdm.get_structure();
[ofdm, rx_time, rx_freq, pn_time, ch_freq] = ofdm.receive(num_signal, random_seed);

result.ref_block_len = [result.ref_block_len; params.ref_block_len];
result.ref_spacing = [result.ref_spacing; params.ref_spacing];
result.ref_density = [result.ref_density; ref_density];

% % LS channel-only algorithm
% ch = LSChOnlyAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym);
% [mu_ch, u_ch, h_ch, nu_ch, v_ch, p_ch] = ch.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_ch, h_ch);
% %ofdm.show_phase_noise_and_channel(1);
% %ofdm.show_constellation(1);
% [ph_nmse_ch, ch_nmse_ch, evm_ch, ber_ch] = ofdm.cal_performance_measure();

% % LS algorithm
% ls_num_iter = 2;
% ls = LSAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ls_num_iter);
% [mu_ls, u_ls, h_ls, nu_ls, v_ls, p_ls] = ls.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_ls, h_ls);
% %ofdm = ofdm.set_perfect_estimated_phase_noise_and_channel();
% %[pn, est_pn_ls, ch, est_ch_ls] = ofdm.show_phase_noise_and_channel(1);
% %ofdm.show_constellation(1);
% [ph_nmse_ls, ch_nmse_ls, evm_ls, ber_ls] = ofdm.cal_performance_measure();
% result.ls.pn_nmse = [result.ls.pn_nmse; ph_nmse_ls];
% result.ls.ch_nmse = [result.ls.ch_nmse; ch_nmse_ls];
% result.ls.ber = [result.ls.ber; ber_ls];
% fprintf('LS algorithm finished.\n');
%   
% % MM algorithm
% mm_num_iter = 256;
% mm = MMAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, mm_num_iter);
% [mu_mm, u_mm, h_mm, nu_mm, v_mm, p_mm] = mm.estimate(rx_freq);
% ofdm = ofdm.set_estimated_phase_noise_and_channel(v_mm, h_mm);
% %[pn, est_pn_mm, ch, est_ch_mm] = ofdm.show_phase_noise_and_channel(1);
% % ofdm.show_constellation(1);
% [ph_nmse_mm, ch_nmse_mm, evm_mm, ber_mm] = ofdm.cal_performance_measure();
% result.mm.pn_nmse = [result.mm.pn_nmse; ph_nmse_mm];
% result.mm.ch_nmse = [result.mm.ch_nmse; ch_nmse_mm];
% result.mm.ber = [result.mm.ber; ber_mm];
% fprintf('MM algorithm finished.\n');

% GLS algorithm
p = 0.99;
R_nu = (1-p)*eye(num_pn_coef) + p*ones(num_pn_coef);
sigma_nh_sq = 0.03;
rho = 1;
gls_num_iter = 2;
gls = GLSAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ...
    R_nu, sigma_nh_sq, rho, gls_num_iter);
[mu_gls, u_gls, h_gls, nu_gls, v_gls, p_gls] = gls.estimate(rx_freq);
ofdm = ofdm.set_estimated_phase_noise_and_channel(v_gls, h_gls);
%[pn, est_pn_gls, ch, est_ch_gls] = ofdm.show_phase_noise_and_channel(1);
% ofdm.show_constellation(1);
[ph_nmse_gls, ch_nmse_gls, evm_gls, ber_gls] = ofdm.cal_performance_measure();
result.gls.pn_nmse = [result.gls.pn_nmse; ph_nmse_gls];
result.gls.ch_nmse = [result.gls.ch_nmse; ch_nmse_gls];
result.gls.ber = [result.gls.ber; ber_gls];
fprintf('GLS algorithm finished.\n');

%LMMSE algorithm
p = 0.99;
R_nu = (1-p)*eye(num_pn_coef) + p*ones(num_pn_coef);
sigma_nh_sq = 0.03;
rho = 1;
lmmse_num_iter = 2;
lmmse = LMMSEAlgorithm(params.fft_size, num_ch_coef, num_pn_coef, ref_mat, ref_sym, ...
    R_nu, sigma_nh_sq, rho, lmmse_num_iter);
[mu_lmmse, u_lmmse, h_lmmse, nu_lmmse, v_lmmse, p_lmmse] = lmmse.estimate(rx_freq);
ofdm = ofdm.set_estimated_phase_noise_and_channel(v_lmmse, h_lmmse);
%[pn, est_pn_lmmse, ch, est_ch_lmmse] = ofdm.show_phase_noise_and_channel(1);
%ofdm.show_constellation(1);
[ph_nmse_lmmse, ch_nmse_lmmse, evm_lmmse, ber_lmmse] = ofdm.cal_performance_measure();
result.lmmse.pn_nmse = [result.lmmse.pn_nmse; ph_nmse_lmmse];
result.lmmse.ch_nmse = [result.lmmse.ch_nmse; ch_nmse_lmmse];
result.lmmse.ber = [result.lmmse.ber; ber_lmmse];
fprintf('LMMSE algorithm finished.\n');

%result.summary = [result.ref_spacing result.ref_block_len result.ref_density result.ls.pn_nmse result.mm.pn_nmse result.gls.pn_nmse result.lmmse.pn_nmse result.ls.ch_nmse result.mm.ch_nmse result.gls.ch_nmse result.lmmse.ch_nmse result.ls.ber result.mm.ber result.gls.ber result.lmmse.ber];
result.summary = [result.ref_spacing result.ref_block_len result.ref_density result.gls.pn_nmse result.lmmse.pn_nmse result.gls.ch_nmse result.lmmse.ch_nmse result.gls.ber result.lmmse.ber];


save result_sbc.mat result

end
end

