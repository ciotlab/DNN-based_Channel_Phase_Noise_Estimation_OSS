function M = circulant(x)
    x = reshape(x, [1 numel(x)]);
    M = toeplitz(x, [x(1) fliplr(x(2:end))]);
end

