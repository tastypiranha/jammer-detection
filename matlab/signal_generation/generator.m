%% Generate Slow BPSK for HackRF
clear; clc;

Fs = 2e6;             % 2 MHz sample rate
bitDuration = 5;      % each bit 5 seconds
totalTime = 30;

bits = repmat([1 0], 1, totalTime/(2*bitDuration));
samplesPerBit = Fs * bitDuration;

fid = fopen('bpsk.iq', 'wb');
fprintf("Generating...\n")

for k = 1:length(bits)
    if bits(k) == 1
        symbol = 127;
    else
        symbol = -127;
    end

    I = int8(symbol * ones(samplesPerBit, 1));
    Q = int8(zeros(samplesPerBit, 1));

    iq = zeros(2*samplesPerBit, 1, 'int8');
    iq(1:2:end) = I;
    iq(2:2:end) = Q;

    fwrite(fid, iq, 'int8');
    fprintf("Bit %d = %d\n", k, bits(k))
end

fclose(fid);
disp("Done")
