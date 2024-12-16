classdef LSAlgorithm < BaseEstimator    
    
    properties
        num_iter
    end
    
    methods
        function obj = LSAlgorithm(N, M, L, T, x_r, num_iter)            
            obj@BaseEstimator(N, M, L, T, x_r);
            obj.num_iter = num_iter;
        end
        
        function [mu, nu] = estimator(obj, y, N, M, L, Q, F, F_mu, T, C, x_r)
            num = size(y, 2);
            mu = zeros(M, num);
            nu = zeros(L, num);
            X_r = diag(x_r);
            mu_const = (1/sqrt(N))*inv(F_mu'*X_r'*C*X_r*F_mu)*F_mu'*X_r'*T';
            for n = 1:num            
                ty = F'*y(:, n);
                tY = diag(ty);  
                nu_const = sqrt(N)*inv(Q'*tY'*F'*C*F*tY*Q)*Q'*tY'*F'*T';
                nu(:, n) = ones(L, 1);
                for i=1:obj.num_iter
                    v = Q*nu(:, n);
                    mu(:, n) = mu_const*T*F*(conj(v).*ty);
                    nu(:, n) = conj(nu_const*T*X_r*F_mu*mu(:, n));                    
                end
            end
        end
    end
end

