classdef LSChOnlyAlgorithm < BaseEstimator    
    
    properties
        num_iter
    end
    
    methods
        function obj = LSChOnlyAlgorithm(N, M, L, T, x_r)            
            obj@BaseEstimator(N, M, L, T, x_r);
        end
        
        function [mu, nu] = estimator(obj, y, N, M, L, Q, F, F_mu, T, C, x_r)
            num = size(y, 2);
            mu = zeros(M, num);
            nu = ones(L, num);
            X_r = diag(x_r);
            mu_const = (1/sqrt(N))*inv(F_mu'*X_r'*C*X_r*F_mu)*F_mu'*X_r'*T';
            for n = 1:num                            
                mu(:, n) = mu_const*T*y(:, n);
            end
        end
    end
end

