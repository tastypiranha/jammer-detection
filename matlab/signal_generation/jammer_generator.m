%% Generate Sinusoidal Jammer for HackRF
clear; clc;

Fs = 2e6;
totalTime = 30;

% Jammer power: 30% of main signal
jammerPower = 0.30;
jammerAmp = round(127 * sqrt(jammerPower));

% Jammer frequency (baseband)
fJammer = 1000;  % 1 kHz tone

fprintf("Jammer amplitude: %d (out of 127)\n", jammerAmp)
fprintf("Power ratio: %.0f%%\n", jammerPower*100)

totalSamples = Fs * totalTime;

fid = fopen('jammer.iq', 'wb');
fprintf("Generating...\n")

blockSize = 1e6;
numBlocks = ceil(totalSamples / blockSize);

for k = 1:numBlocks
    startSample = (k-1)*blockSize;
    n = min(blockSize, totalSamples - startSample);
    t = ((startSample):(startSample + n - 1))' / Fs;

    I = int8(jammerAmp * cos(2*pi*fJammer*t));
    Q = int8(zeros(n, 1));

    iq = zeros(2*n, 1, 'int8');
    iq(1:2:end) = I;
    iq(2:2:end) = Q;

    fwrite(fid, iq, 'int8');
    fprintf("Block %d/%d\n", k, numBlocks)
end

fclose(fid);
fprintf("Done!\n")
fprintf("hackrf_transfer -t jammer.iq -f 915000000 -s 2000000 -x 30 -a 0\n")
