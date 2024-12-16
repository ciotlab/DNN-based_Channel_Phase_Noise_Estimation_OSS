classdef GLSAlgorithm < BaseEstimator    
    
    properties
        R_mu
        R_nu
        sigma_nh_sq
        rho
        num_iter        
        M_nu
    end
    
    methods
        function obj = GLSAlgorithm(N, M, L, T, x_r, R_nu, sigma_nh_sq, rho, num_iter)            
            obj@BaseEstimator(N, M, L, T, x_r);
            obj.R_nu = R_nu; % Covariance matrix of nu
            obj.sigma_nh_sq = sigma_nh_sq; % Noise variance * Variance of 1/h
            obj.rho = rho; % Expected power of a data symbol.
            obj.num_iter = num_iter;    
            Q = obj.Q;
            C = obj.C;
            F = obj.F;
            Psi_r = circulant(x_r);
            Ohm = sigma_nh_sq*eye(size(T, 1));
            D = find(1-diag(C));
            R = find(diag(C));
            tmp = (rho/N)*F*Q*R_nu*Q'*F';
            for i = 1:numel(D)
                n = D(i);
                idx = mod(R-(n-1)-1, N)+1;
                Ohm = Ohm + tmp(idx, idx);
            end
            inv_Ohm = inv(Ohm);
            tmp = T*Psi_r*F*Q;
            obj.M_nu = sqrt(N)*inv(tmp'*inv_Ohm*tmp)*tmp'*inv_Ohm;
        end
        
        function [mu, nu] = estimator(obj, y, N, M, L, Q, F, F_mu, T, C, x_r)            
            num = size(y, 2);            
            mu = zeros(M, num);
            nu = zeros(L, num);
            X_r = diag(x_r);
            mu_const = (1/sqrt(N))*inv(F_mu'*X_r'*C*X_r*F_mu)*F_mu'*X_r'*T';
            for n = 1:num 
                ty = F'*y(:, n);
                nu(:, n) = ones(L, 1);
                for i = 1:obj.num_iter   
                    v = Q*nu(:, n);
                    mu(:, n) = mu_const*T*F*(conj(v).*ty);
                    h = sqrt(N)*F_mu*mu(:, n);
                    z = T*(y(:, n)./h);
                    nu(:, n) = obj.M_nu*z;
                end
            end
        end
    end
end

