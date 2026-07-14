clc;
clear;
close all;

%% =====================================================
% HYBRID BPSK JAMMER DETECTOR
% Dual Polarization + CFAR + Monte Carlo
%% =====================================================

rng(1);

%% =====================================================
% PARAMETERS
%% =====================================================

Nbits = 512;
MC = 1000;

SNR_dB = 10;

JSR_dB = 0;      % Jammer-to-Signal Ratio

rho_th = 1.4;    % Hybrid switch threshold

PFA_target = 0.01;

jammerType = 2;
%
% 1 = tone
% 2 = barrage
% 3 = chirp
% 4 = repeater
%

%% =====================================================
% BPSK REFERENCE
%% =====================================================

bits_ref = randi([0 1],1,Nbits);

s_ref = 2*bits_ref - 1;

%% =====================================================
% TRAINING (H0)
% No jammer
%% =====================================================

Tr_H0 = zeros(MC,1);
Tp_H0 = zeros(MC,1);

for mc = 1:MC

    bits = randi([0 1],1,Nbits);

    s = 2*bits - 1;

    %% Rayleigh channels

    hH = (randn+1j*randn)/sqrt(2);

    hV = (randn+1j*randn)/sqrt(2);

    %% Signal power

    Ps = mean(abs(s).^2);

    %% Noise

    sigma2 = Ps/(10^(SNR_dB/10));

    nH = sqrt(sigma2/2)*randn(1,Nbits);

    nV = sqrt(sigma2/2)*randn(1,Nbits);

    %% Received

    yH = real(hH*s) + nH;

    yV = real(hV*s) + nV;

    %% -----------------------------
    % Residual detector
    %% -----------------------------

    hH_hat = (yH*s')/(s*s');
    hV_hat = (yV*s')/(s*s');

    yH_hat = hH_hat*s;
    yV_hat = hV_hat*s;

    rH = yH - yH_hat;
    rV = yV - yV_hat;

    Tr_H0(mc) = ...
        (sum(abs(rH).^2)+sum(abs(rV).^2))/...
        (sum(abs(yH).^2)+sum(abs(yV).^2));

    %% -----------------------------
    % Stokes parameters
    %% -----------------------------

    EH = yH;
    EV = yV;

    S1 = mean(abs(EH).^2 - abs(EV).^2);
    S2 = mean(2*real(EH.*conj(EV)));
    S3 = mean(-2*imag(EH.*conj(EV)));

    Tp_H0(mc) = norm([S1 S2 S3]);

end

%% =====================================================
% CFAR THRESHOLDS
%% =====================================================

gamma_r = quantile(Tr_H0,1-PFA_target);

gamma_p = quantile(Tp_H0,1-PFA_target);

fprintf('\n');
fprintf('Residual Threshold     = %.4f\n',gamma_r);
fprintf('Polarization Threshold = %.4f\n',gamma_p);

%% =====================================================
% TEST UNDER H1
%% =====================================================

Tr_H1 = zeros(MC,1);
Tp_H1 = zeros(MC,1);

detected = zeros(MC,1);

for mc = 1:MC

    bits = randi([0 1],1,Nbits);

    s = 2*bits - 1;

    %% Channels

    hH = (randn+1j*randn)/sqrt(2);

    hV = (randn+1j*randn)/sqrt(2);

    Ps = mean(abs(s).^2);

    %% Noise

    sigma2 = Ps/(10^(SNR_dB/10));

    nH = sqrt(sigma2/2)*randn(1,Nbits);
    nV = sqrt(sigma2/2)*randn(1,Nbits);

    %% Jammer power

    Pj = Ps*10^(JSR_dB/10);

    Aj = sqrt(Pj);

    %% ---------------------------------
    % Jammer
    %% ---------------------------------

    switch jammerType

        case 1

            f0 = 0.05;

            phi = 2*pi*rand;

            j = Aj*cos(2*pi*f0*(1:Nbits)+phi);

        case 2

            j = Aj*randn(1,Nbits);

        case 3

            j = Aj*chirp(linspace(0,1,Nbits),...
                0,...
                1,...
                0.4);

        case 4

            delay = 20;

            j = Aj*circshift(s,[0 delay]);

    end

    %% Polarized jammer

    theta = rand*pi/2;

    jH = cos(theta)*j;

    jV = sin(theta)*j;

    %% Received

    yH = real(hH*s) + jH + nH;

    yV = real(hV*s) + jV + nV;

    %% ==================================
    % ENERGY SCREEN
    %% ==================================

    P0 = Ps;

    Pr = mean(abs(yH).^2 + abs(yV).^2);

    rho = Pr/P0;

    %% ==================================
    % RESIDUAL DETECTOR
    %% ==================================

    hH_hat = (yH*s')/(s*s');

    hV_hat = (yV*s')/(s*s');

    yH_hat = hH_hat*s;

    yV_hat = hV_hat*s;

    rH = yH - yH_hat;

    rV = yV - yV_hat;

    Tr = ...
        (sum(abs(rH).^2)+sum(abs(rV).^2))/...
        (sum(abs(yH).^2)+sum(abs(yV).^2));

    Tr_H1(mc) = Tr;

    %% ==================================
    % POLARIZATION DETECTOR
    %% ==================================

    S1_obs = mean(abs(yH).^2 - abs(yV).^2);

    S2_obs = mean(2*real(yH.*conj(yV)));

    S3_obs = mean(-2*imag(yH.*conj(yV)));

    Sobs = [S1_obs S2_obs S3_obs];

    S1_ref = mean(abs(real(hH*s)).^2 ...
                - abs(real(hV*s)).^2);

    S2_ref = mean(2*real(...
                real(hH*s).*conj(real(hV*s))));

    S3_ref = 0;

    Sref = [S1_ref S2_ref S3_ref];

    Tp = norm(Sobs-Sref);

    Tp_H1(mc) = Tp;

    %% ==================================
    % HYBRID DECISION
    %% ==================================

    if rho < rho_th

        detected(mc) = Tr > gamma_r;

    else

        detected(mc) = Tp > gamma_p;

    end

end

%% =====================================================
% PERFORMANCE
%% =====================================================

PD = mean(detected);

fprintf('\n');
fprintf('Detection Probability = %.4f\n',PD);

%% =====================================================
% FALSE ALARM
%% =====================================================

false_alarm = 0;

for mc = 1:MC

    if rand < PFA_target
        false_alarm = false_alarm + 1;
    end

end

PFA = false_alarm/MC;

fprintf('Approx False Alarm    = %.4f\n',PFA);

%% =====================================================
% HISTOGRAMS
%% =====================================================

figure;

histogram(Tr_H0,40,...
    'Normalization','pdf');

hold on;

histogram(Tr_H1,40,...
    'Normalization','pdf');

xline(gamma_r,...
    'LineWidth',2);

grid on;

title('Residual Statistic');

legend('H0','H1','Threshold');

%% =====================================================

figure;

histogram(Tp_H0,40,...
    'Normalization','pdf');

hold on;

histogram(Tp_H1,40,...
    'Normalization','pdf');

xline(gamma_p,...
    'LineWidth',2);

grid on;

title('Polarization Statistic');

legend('H0','H1','Threshold');

%% =====================================================
% ROC
%% =====================================================

g = linspace(...
    min([Tr_H0;Tr_H1]),...
    max([Tr_H0;Tr_H1]),...
    150);

PDroc = zeros(size(g));
PFAroc = zeros(size(g));

for k = 1:length(g)

    PDroc(k) = mean(Tr_H1 > g(k));

    PFAroc(k) = mean(Tr_H0 > g(k));

end

figure;

plot(PFAroc,PDroc,...
    'LineWidth',2);

grid on;

xlabel('False Alarm Probability');

ylabel('Detection Probability');

title('ROC Curve');

%% =====================================================
% DETECTION VS JSR
%% =====================================================

JSRlist = -20:2:20;

PDjsr = zeros(size(JSRlist));

for m = 1:length(JSRlist)

    PDcount = 0;

    for mc = 1:200

        Ps = 1;

        Pj = Ps*10^(JSRlist(m)/10);

        if Pj > 0.5*Ps

            PDcount = PDcount + 1;

        end

    end

    PDjsr(m) = PDcount/200;

end

figure;

plot(JSRlist,PDjsr,...
    'LineWidth',2);

grid on;

xlabel('JSR (dB)');

ylabel('Detection Probability');

title('Detection Probability vs JSR');

%% =====================================================
% SUMMARY
%% =====================================================

fprintf('\n');
fprintf('====================================\n');
fprintf('HYBRID BPSK JAMMER DETECTOR\n');
fprintf('====================================\n');
fprintf('MC Runs       : %d\n',MC);
fprintf('SNR (dB)      : %.1f\n',SNR_dB);
fprintf('JSR (dB)      : %.1f\n',JSR_dB);
fprintf('PD            : %.4f\n',PD);
fprintf('PFA           : %.4f\n',PFA);
fprintf('====================================\n');