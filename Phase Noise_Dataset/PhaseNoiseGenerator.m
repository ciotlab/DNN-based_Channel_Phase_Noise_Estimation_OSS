classdef PhaseNoiseGenerator

    properties
        sample_rate        
        carrier_freq
        fft_size
        freq_conv_mat
        model
        psd
        phase_noise
    end

    methods
        function obj = PhaseNoiseGenerator(params)
            obj.model = params.pn_model;
            obj.sample_rate = params.sample_rate; %Hz
            obj.carrier_freq = params.carrier_freq; %Hz
            obj.fft_size = params.fft_size;
            obj.freq_conv_mat = fft_matrix(obj.fft_size)/sqrt(obj.fft_size);
            % Pole/zeros and PSD0
            switch obj.model
                case 'A'
                    % Parameter set A (R1-163984)
                    fc_base = 30e9;
                    fz = [1.8 2.2 40]*1e6;
                    fp = [0.1 0.2 8]*1e6;
                    alphaz = [2 2 2];
                    alphap = [2 2 2];
                    PSD0 = -79.4;
                case 'B'
                    % Parameter set B (R1-163984)
                    fc_base = 60e9;
                    fz = [0.02 6 10]*1e6;
                    fp = [0.005 0.4 0.6]*1e6;
                    alphaz = [2 2 2];
                    alphap = [2 2 2];
                    PSD0 = -70;
                case 'C'
                    % Parameter set C (TR 38.803)
                    fc_base = 29.55e9;
                    fz = [3e3 550e3 280e6];
                    fp = [1 1.6e6 30e6];
                    alphaz = [2.37 2.7 2.53];
                    alphap = [3.3 3.3 1];
                    PSD0 = 32;
            end
            foffset_log = (4:0.1:log10(obj.sample_rate/2));
            foffset = 10.^foffset_log;
            num = ones(size(foffset));
            for ii = 1:numel(fz)
                num = num.*(1 + (foffset./fz(ii)).^alphaz(ii));
            end
            den = ones(size(foffset));
            for ii = 1:numel(fp)
                den = den.*(1 + (foffset./fp(ii)).^alphap(ii));
            end
            obj.psd = 10*log10(num./den) + PSD0 + 20*log10(obj.carrier_freq/fc_base);             
            obj.phase_noise = comm.PhaseNoise('FrequencyOffset', foffset, 'Level', obj.psd, 'SampleRate', obj.sample_rate);
            sig = ones(obj.fft_size, 1);
            while all(sig == 1)
                sig = obj.phase_noise(ones(obj.fft_size, 1));
            end
            for n = 1:16
                obj.phase_noise(ones(obj.fft_size, 1));
            end
        end
        
        function pn_time = get(obj, num)
            pn_time = zeros(obj.fft_size, num);
            %pn_time_mat = zeros(obj.fft_size, obj.fft_size, num);
            %pn_freq = zeros(obj.fft_size, num);
            %pn_freq_mat = zeros(obj.fft_size, obj.fft_size, num);
            for n = 1:num                
                pn = obj.phase_noise(ones(obj.fft_size, 1));
                pn_time(:, n) = pn;
                %pn_time_mat(:, :, n) = diag(pn);
                %pn_freq(:, n) = obj.freq_conv_mat * pn;
                %pn_freq_mat(:, :, n) = circulant(pn_freq(:, n));
            end
        end
    end
end

