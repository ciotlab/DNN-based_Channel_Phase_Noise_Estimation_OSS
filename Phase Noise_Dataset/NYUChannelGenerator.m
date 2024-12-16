classdef NYUChannelGenerator

    properties
        carrier_freq
        sample_rate
        fft_size   
        cp_len
        sample_interval
        sample_time
        freq_conv_mat
        pdp
    end

    methods
        function obj = NYUChannelGenerator(params)
            obj.carrier_freq = params.carrier_freq; % carrier frequency in Hz
            obj.sample_rate = params.sample_rate; % Hz                        
            obj.fft_size = params.fft_size;
            obj.cp_len = params.cp_len;
            obj.sample_interval = 1/obj.sample_rate * (10^9); % ns
            obj.sample_time = linspace(0, obj.fft_size-1, obj.fft_size);
            obj.freq_conv_mat = fft_matrix(obj.fft_size)*sqrt(obj.fft_size);                   
            obj.pdp = load(params.pdp_file).pdp; % Load power delay profile            
        end
        
        function ch_time = get(obj, num)            
            ch_time = zeros(obj.fft_size, num);
            %ch_time_mat = zeros(obj.fft_size, obj.fft_size, num);
            %ch_freq = zeros(obj.fft_size, num);
            %ch_freq_mat = zeros(obj.fft_size, obj.fft_size, num);
            for n = 1:num
                p = obj.pdp{randi(numel(obj.pdp))};
                cp_lim = repmat(p(:, 1) < obj.cp_len, [1 2]);
                p = p(cp_lim);
                p = reshape(p, [numel(p)/2 2]);
                num_taps = size(p, 1);
                delay = p(:, 1)/obj.sample_interval;                
                time_diff = repmat(obj.sample_time, [num_taps 1]) - repmat(delay, [1 obj.fft_size]);
                sampled_pdp = sinc(time_diff);
                mag = sqrt(p(:, 2)).';
                coef = ((randn(1, num_taps) + 1i*randn(1, num_taps))/sqrt(2)) .* mag;
                ch = (coef * sampled_pdp).';
                ch = ch / sqrt(sum(abs(ch).^2));
                ch_time(:, n) = ch;
                %ch_time_mat(:, :, n) = circulant(ch_time(:, n));
                %ch_freq(:, n) = obj.freq_conv_mat * ch;
                %ch_freq_mat(:, :, n) = diag(ch_freq(:, n));
            end
        end
    end
end

