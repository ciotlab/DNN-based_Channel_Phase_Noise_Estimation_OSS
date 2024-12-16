num_pdp = 1000;
pdp = cell(1, num_pdp);

for k = 1:num_pdp
    file_name = sprintf('PDP_mmWave/OmniPDP%d_CoPol.mat', k);
    raw_pdp = load(file_name).OmniPDP;
    num_taps = size(raw_pdp, 1);
    start_delay = raw_pdp(1, 1);    
    for t = 1:num_taps
        delay = raw_pdp(t, 1);                
        delay = delay - start_delay;  
        raw_pdp(t, 1) = delay; % ns
        pwr_db = raw_pdp(t, 2);
        pwr = 10^(pwr_db/10); 
        raw_pdp(t, 2) = pwr; % power
    end
    pdp{k} = raw_pdp;
end

save PDP_mmWave.mat pdp