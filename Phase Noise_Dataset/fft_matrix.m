function F = fft_matrix(K)
    F = exp(-1i*2*pi*((0:K-1).'*(0:K-1))/K)/sqrt(K);
end

