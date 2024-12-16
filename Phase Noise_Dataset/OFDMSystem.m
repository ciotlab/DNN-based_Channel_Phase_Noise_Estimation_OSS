classdef OFDMSystem
    
    properties
        fft_size
        data_mod_order
        fft_mat
        pn_time_pool
        ch_freq_pool
        num_ref_sym
        ref_loc
        ref_mat
        ref_sym
        data_loc
        num_data_sym
        snr_db
        tx_time
        tx_freq
        tx_data
        rx_time
        rx_freq
        ch_freq
        pn_time
        pn_common_phase        
        rx_sig
        noise
        est_pn_time
        est_pn_common_phase
        est_ch_freq
        corr_rx_time
        corr_rx_freq
        tdl_seed_base
    end
    
    methods
        function obj = OFDMSystem(params)            
            obj.fft_size = params.fft_size;
            obj.data_mod_order = params.data_mod_order;
            obj.fft_mat = fft_matrix(obj.fft_size);
            c = load('channel.mat');
            obj.pn_time_pool = c.pn_time;
            obj.ch_freq_pool = c.ch_freq;
            obj.ref_loc = false(obj.fft_size, 1); 
            if strcmp(params.ref_type, 'SBC') % SBC: single block and comb 
                obj.ref_loc(1:1:params.ref_block_len) = true;
                obj.ref_loc(1:params.ref_spacing:end) = true;            
            elseif strcmp(params.ref_type, 'MB') % MB: multi-block
                for first = 1:params.ref_block_len
                    obj.ref_loc(first:params.ref_spacing:end) = true;
                end
            end
            obj.num_ref_sym = sum(obj.ref_loc);            
            obj.ref_mat = zeros(obj.num_ref_sym, obj.fft_size);
            ref_cnt = 1;
            for i = 1:obj.fft_size
                if obj.ref_loc(i)
                    obj.ref_mat(ref_cnt, i) = 1;
                    ref_cnt = ref_cnt + 1;
                end
            end
            x = randi([0 3], obj.num_ref_sym, 1);
            obj.ref_sym = zeros(obj.fft_size, 1);
            obj.ref_sym(obj.ref_loc) = qammod(x, 4, 'UnitAveragePower', true);
            obj.data_loc = ~obj.ref_loc;
            obj.num_data_sym = sum(obj.data_loc);
            obj.snr_db = params.snr_db;
        end

        function [ref_sym, ref_loc, ref_mat, data_loc, ref_density] = get_structure(obj)
            ref_sym = obj.ref_sym;
            ref_loc = obj.ref_loc;
            ref_mat = obj.ref_mat;
            data_loc = obj.data_loc;
            ref_density = obj.num_ref_sym/obj.fft_size*100;
        end
        
        function [time_sym, freq_sym, data_bit] = transmit(obj, num)
            time_sym = zeros(obj.fft_size, num);
            freq_sym = zeros(obj.fft_size, num);
            data_bit = zeros(obj.num_data_sym*log2(obj.data_mod_order), num);
            for n = 1:num
                x = randi([0 obj.data_mod_order-1], obj.num_data_sym, 1);
                data_sym = qammod(x, obj.data_mod_order, 'UnitAveragePower', true);
                fs = obj.ref_sym;
                fs(obj.data_loc) = data_sym;
                freq_sym(:, n) = fs;
                time_sym(:, n) = obj.fft_mat' * fs;
                if ~isempty(x)
                    data_bit(:, n) = int2bit(x, log2(obj.data_mod_order));
                end
            end
        end

        function [pn_time, ch_freq] = channel(obj, num)
            pn_time = zeros(obj.fft_size, num);
            ch_freq = zeros(obj.fft_size, num);
            for n = 1:num
                pn_time(:, n) = obj.pn_time_pool(:, randi(size(obj.pn_time_pool, 2)));
                ch_freq(:, n) = obj.ch_freq_pool(:, randi(size(obj.ch_freq_pool, 2)));
            end
        end

        function [obj, rx_time, rx_freq, pn_time, ch_freq] = receive(obj, num, seed)
            rng(seed);
            [obj.tx_time, obj.tx_freq, obj.tx_data] = obj.transmit(num);
            rng(seed);
            [pn_time, ch_freq] = obj.channel(num);                       
            noise_pow = db2pow(-obj.snr_db);

            obj.noise = sqrt(noise_pow/2) * (randn(obj.fft_size, num) + 1i*randn(obj.fft_size, num));
            obj.rx_time = pn_time .* (obj.fft_mat' * (ch_freq .* obj.tx_freq)) + obj.noise;
            obj.rx_freq = obj.fft_mat * obj.rx_time;
            obj.pn_time = pn_time;
            obj.ch_freq = ch_freq;
            common_phase = repmat(angle(sum(obj.pn_time, 1)), [obj.fft_size 1]);
            obj.pn_time = obj.pn_time .* exp(-1i*common_phase);
            obj.ch_freq = obj.ch_freq .* exp(1i*common_phase);

            rx_time = obj.rx_time;
            rx_freq = obj.rx_freq;
            obj.est_pn_time = ones(obj.fft_size, num);
            obj.est_ch_freq = ones(obj.fft_size, num);          
        end

        function obj = set_estimated_phase_noise_and_channel(obj, est_pn_time, est_ch_freq)
            obj.est_pn_time = est_pn_time;
            obj.est_ch_freq = est_ch_freq; 
            common_phase = repmat(angle(sum(obj.est_pn_time, 1)), [obj.fft_size 1]);
            obj.est_pn_time = obj.est_pn_time .* exp(-1i*common_phase);
            obj.est_ch_freq = obj.est_ch_freq .* exp(1i*common_phase);
            obj = obj.correct_rx_signal();
        end

        function obj = set_perfect_estimated_phase_noise_and_channel(obj)
            obj.est_pn_time = obj.pn_time;            
            obj.est_ch_freq = obj.ch_freq; 
            obj = obj.correct_rx_signal();
        end

        function obj = correct_rx_signal(obj)
            %pn_corrected = exp(-1i*angle(obj.est_pn_time)) .* obj.rx_time;
            pn_corrected_time = obj.rx_time./obj.est_pn_time;
            pn_corrected_freq = obj.fft_mat * pn_corrected_time;
            obj.corr_rx_freq = pn_corrected_freq./obj.est_ch_freq;
            obj.corr_rx_time = obj.fft_mat' * obj.corr_rx_freq;
        end

        function [pn, est_pn, ch, est_ch] = show_phase_noise_and_channel(obj, ind)
            x = 1:obj.fft_size;
            pn = obj.pn_time(:, ind);
            est_pn = obj.est_pn_time(:, ind);
            ch = obj.ch_freq(:, ind);
            est_ch = obj.est_ch_freq(:, ind);
            subplot(2,2,1)
            plot(x, abs(pn), x, abs(est_pn));
            title('Phase noise (magnitude)')
            subplot(2,2,2)
            plot(x, angle(pn), x, angle(est_pn));
            title('Phase noise (angle)')
            subplot(2,2,3)
            plot(x, abs(ch), x, abs(est_ch));
            title('Channel (magnitude)')
            subplot(2,2,4)
            plot(x, angle(ch), x, angle(est_ch));
            title('Channel (angle)')
        end

        function [tx_data_sym, rx_data_sym] = show_constellation(obj, ind)
            tx_data_sym = obj.tx_freq(obj.data_loc, ind);
            rx_data_sym = obj.corr_rx_freq(obj.data_loc, ind);
            scatterplot(rx_data_sym)
        end      

        function [ph_nmse, ch_nmse, evm, ber] = cal_performance_measure(obj)
            pn = angle(obj.pn_time);
            est_pn = angle(obj.est_pn_time);
            ph_nmse = sum(sum(abs(pn - est_pn).^2))/sum(sum(abs(pn).^2));
            ch = obj.ch_freq;
            est_ch = obj.est_ch_freq;
            ch_nmse = sum(sum(abs(ch - est_ch).^2))/sum(sum(abs(ch).^2));
            tx_data_sym = obj.tx_freq(obj.data_loc, :);
            rx_data_sym = obj.corr_rx_freq(obj.data_loc, :);
            evm = sqrt(sum(sum(abs(tx_data_sym - rx_data_sym).^2))/sum(sum(abs(tx_data_sym).^2)))*100;
            rx_data_sym_flat = reshape(rx_data_sym, [], 1);
            rx_data = qamdemod(rx_data_sym_flat, obj.data_mod_order, 'UnitAveragePower', true);
            rx_data = int2bit(rx_data, log2(obj.data_mod_order));
            tx_data_flat = reshape(obj.tx_data, [], 1);
            [~, ber] = biterr(tx_data_flat, rx_data);
        end
    end
end

