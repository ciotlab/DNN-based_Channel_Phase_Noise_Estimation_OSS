classdef TDLChannelGenerator

    properties
        carrier_freq
        sample_rate
        fft_size
        tdl
        freq_conv_mat
    end

    methods
        function obj = TDLChannelGenerator(params)
            obj.carrier_freq = params.carrier_freq; % carrier frequency in Hz
            obj.sample_rate = params.sample_rate; % Hz
            obj.fft_size = params.fft_size;
            obj.freq_conv_mat = fft_matrix(obj.fft_size)*sqrt(obj.fft_size);

            % TDL channel
            obj.tdl = nrTDLChannel;
            obj.tdl.DelayProfile = params.delay_profile;
            obj.tdl.DelaySpread = params.delay_spread;
            obj.tdl.MaximumDopplerShift = 0;
            obj.tdl.NumTransmitAntennas = 1;
            obj.tdl.NumReceiveAntennas = 1;
            obj.tdl.SampleRate = obj.sample_rate;
        end
        
        function [ch_time, ch_time_mat, ch_freq, ch_freq_mat] = get(obj, num, seed_base)
            %seed_base = floor(sum(100*clock));
            ch_time = zeros(obj.fft_size, num);
            ch_time_mat = zeros(obj.fft_size, obj.fft_size, num);
            ch_freq = zeros(obj.fft_size, num);
            ch_freq_mat = zeros(obj.fft_size, obj.fft_size, num);
            for n = 1:num
                obj.tdl.release();
                obj.tdl.Seed = mod(seed_base + n, 2^32);
                tx_sig = zeros(obj.fft_size, 1);
                tx_sig(1) = 1;
                rx_sig = obj.tdl(tx_sig);
                ch_time(:, n) = rx_sig;
                ch_time_mat(:, :, n) = circulant(ch_time(:, n));
                ch_freq(:, n) = obj.freq_conv_mat * rx_sig;
                ch_freq_mat(:, :, n) = diag(ch_freq(:, n));
            end
        end
    end
end

