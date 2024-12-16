classdef MMAlgorithm < BaseEstimator    
    
    properties
        num_iter
    end
    
    methods
        function obj = MMAlgorithm(N, M, L, T, x_r, num_iter)            
            obj@BaseEstimator(N, M, L, T, x_r);
            obj.num_iter = num_iter;
        end
        
        function [mu, nu] = estimator(obj, y, N, M, L, Q, F, F_mu, T, C, x_r)
            num = size(y, 2);
            X_r = diag(x_r);
            Wconst = (eye(size(T, 1))-T*X_r*F_mu*inv(F_mu'*X_r'*C*X_r*F_mu)*F_mu'*X_r'*T');
            mu_const = (1/sqrt(N))*inv(F_mu'*X_r'*C*X_r*F_mu)*F_mu'*X_r'*T';
            mu = zeros(M, num);
            nu = ones(L, num);
            for n = 1:num
                Y = circulant(y(:, n));  
                W = Wconst*T*Y*F*Q;
                U = W.'*conj(W);
                lambda = trace(U);
                for i = 1:obj.num_iter
                    nu(:, n) = exp(1i*angle((lambda*eye(L)-U)*nu(:, n)));
                end
                p = (1/sqrt(N))*F*Q*nu(:, n);
                P = circulant(p);
                mu(:, n) = mu_const*T*P'*y(:, n);
            end
        end
    end
end

