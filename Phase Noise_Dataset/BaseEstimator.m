classdef BaseEstimator    
    
    properties
        N % Number of subcarriers
        M % Number of channel coefficient
        L % Number of phase noise coefficient
        Q % Interpolation matrix
        F % FFT matrix
        F_mu % Partial FFT matrix
        T % Reference allocation matrix
        C
        x_r % Reference symbol vector
    end
    
    methods
        function obj = BaseEstimator(N, M, L, T, x_r)
            obj.N = N;
            obj.M = M;
            obj.L = L;
            %obj.Q = obj.linear_interpolation_matrix(obj.N, obj.L);
            obj.Q = obj.trigonometric_interpolation_matrix(obj.N, obj.L);
            obj.F = fft_matrix(obj.N);
            obj.F_mu = obj.F(:, 1:obj.M);
            obj.T = T;
            obj.C = obj.T.'*obj.T;
            obj.x_r = x_r;
        end        

        function [mu, u, h, nu, v, p] = estimate(obj, y)
            % mu: low-dimensional time-domain channel
            % u: Time-domain channel
            % h: Frequency-domain channel
            % nu: low-dimensional time-domain phase noise
            % v: Time-domain phase noise
            % p: Frequency-domain phase noise
            [mu, nu] = obj.estimator(y, obj.N, obj.M, obj.L, obj.Q, obj.F, obj.F_mu, obj.T, obj.C, obj.x_r);
            num = size(mu, 2);
            u = zeros(obj.N, num);
            h = zeros(obj.N, num);
            v = zeros(obj.N, num);
            p = zeros(obj.N, num);
            for n = 1:num
                u(1:obj.M, n) = mu(:, n);
                h(:, n) = sqrt(obj.N)*obj.F_mu*mu(:, n);
                v(:, n) = obj.Q*nu(:, n);
                p(:, n) = (1/sqrt(obj.N))*obj.F*v(:, n);
            end
        end
    end

    methods (Abstract)
        estimator(obj, y, N, M, L, Q, F, F_mu, T, C, x_r)
    end

    methods(Static)
        function Q = linear_interpolation_matrix(N, L)
            Q = zeros(N, L);
            Xi = (L-1)/(N-1);
            for n = 1:N
                for l = 1:L
                    if ((l-1)/Xi <= n-1) && (n-1 < l/Xi)
                        Q(n, l) = l-(n-1)*Xi;
                    elseif ((l-2)/Xi <= n-1) && (n-1 < (l-1)/Xi)
                        Q(n, l) = (n-1)*Xi-(l-2);
                    else
                        Q(n, l) = 0;
                    end
                end
            end
        end
        function Q = trigonometric_interpolation_matrix(N, L)
            x_offset = repmat(2*pi*((0:L-1)/L), [N 1]);
            x = repmat(2*pi*((L-1)/L)*((0:N-1).'/(N-1)), [1 L]);
            Q = sinc(L*(x-x_offset)/(2*pi)) ./ sinc((x-x_offset)/(2*pi));
            if mod(L,2) == 0                     
                Q = Q .* cos((x-x_offset)/2); 
            end          
        end
    end
end

