clc;
clear;
close all;

%% ==========================================================
% PHASE-POLARIZATION CONSISTENCY DETECTOR EVALUATION
%
% Tests:
% 1) H0 vs H1 distributions
% 2) ROC curve
% 3) PD vs JSR
%
% Statistic:
%
% u_bar = mean( abs(Qobs_dB - Qexp_dB) )
%
%% ==========================================================

rng(1);

%% ==========================================================
% PARAMETERS
%% ==========================================================

MC = 5000;          % Monte Carlo trials

Nphi = 360;         % phase sweep points

alpha = 0.5;

beta_deg = 40;
beta = deg2rad(beta_deg);

sigma = 0.02;

PFA_target = 0.01;

dphi_deg = linspace(0,360,Nphi);

dphi = deg2rad(dphi_deg);

eps0 = 1e-12;

%% ==========================================================
% STORAGE
%% ==========================================================

u_H0 = zeros(MC,1);

u_H1 = zeros(MC,1);

%% ==========================================================
% H0 : NO ADVERSARY
%% ==========================================================

fprintf('Running H0 trials...\n');

for mc = 1:MC

    Qe = zeros(1,Nphi);
    Qo = zeros(1,Nphi);

    for k = 1:Nphi

        %% Legitimate signal

        EH = alpha*sin(beta)*exp(1j*dphi(k));

        EV = 1 + alpha*cos(beta)*exp(1j*dphi(k));

        %% Expected

        Qe(k) = abs(EV)^2/(abs(EH)^2 + eps0);

        %% Observation with noise

        EHobs = EH + sigma*(randn+1j*randn);

        EVobs = EV + sigma*(randn+1j*randn);

        Qo(k) = abs(EVobs)^2/(abs(EHobs)^2 + eps0);

    end

    %% dB statistic

    ue = abs(10*log10(Qo) - 10*log10(Qe));

    u_H0(mc) = mean(ue);

end

%% ==========================================================
% H1 : ADVERSARY
%% ==========================================================

%% ==========================================================
% H1 : MATCHED-POLARIZATION ADVERSARY
%% ==========================================================

fprintf('Running H1 trials...\n');

JSR_dB = 0;

for mc = 1:MC

    k = 10^(JSR_dB/20);

    Qe = zeros(1,Nphi);
    Qo = zeros(1,Nphi);

    for idx = 1:Nphi

        %% Legitimate channel

        EH = alpha*sin(beta)*exp(1j*dphi(idx));

        EV = 1 + alpha*cos(beta)*exp(1j*dphi(idx));

        %% Expected ratio

        Qe(idx) = abs(EV)^2/(abs(EH)^2 + eps0);

        %% --------------------------------------------------
        % SMART ADVERSARY
        %
        % Jammer exactly follows legitimate polarization
        %% --------------------------------------------------

        EHj = k*EH;

        EVj = k*EV;

        %% Observation

        EHobs = EH + EHj ...
            + sigma*(randn+1j*randn);

        EVobs = EV + EVj ...
            + sigma*(randn+1j*randn);

        Qo(idx) = abs(EVobs)^2/(abs(EHobs)^2 + eps0);

    end

    ue = abs(10*log10(Qo) - 10*log10(Qe));

    u_H1(mc) = mean(ue);

end

%% ==========================================================
% THRESHOLD
%% ==========================================================

gamma = quantile(u_H0,1-PFA_target);

PD = mean(u_H1 > gamma);

PFA = mean(u_H0 > gamma);

fprintf('\n');
fprintf('Threshold = %.4f\n',gamma);
fprintf('PD        = %.4f\n',PD);
fprintf('PFA       = %.4f\n',PFA);

%% ==========================================================
% HISTOGRAM
%% ==========================================================

figure;

histogram(u_H0,60,...
    'Normalization','pdf');

hold on;

histogram(u_H1,60,...
    'Normalization','pdf');

xline(gamma,...
    'LineWidth',2);

grid on;

xlabel('ū');

ylabel('PDF');

title('H0 vs H1 Distribution');

legend('H0','H1','Threshold');

%% ==========================================================
% ROC CURVE
%% ==========================================================

glist = linspace( ...
    min([u_H0;u_H1]), ...
    max([u_H0;u_H1]), ...
    250);

PDroc = zeros(size(glist));
PFAroc = zeros(size(glist));

for k = 1:length(glist)

    g = glist(k);

    PDroc(k) = mean(u_H1 > g);

    PFAroc(k) = mean(u_H0 > g);

end

figure;

plot(PFAroc,PDroc,...
    'LineWidth',2);

hold on;

plot([0 1],[0 1],'--');

grid on;

xlabel('PFA');

ylabel('PD');

title('ROC Curve');

legend('Detector','Random Guess');

%% ==========================================================
% PD VS JSR
%% ==========================================================

fprintf('\nRunning JSR sweep...\n');

JSRlist = -20:2:20;

PDjsr = zeros(size(JSRlist));

for m = 1:length(JSRlist)

    JSR_dB = JSRlist(m);

    detections = 0;

    MCjsr = 1000;

    Aj = 10^(JSR_dB/20);

    for mc = 1:MCjsr

        theta = rand*pi/2;

        phi = 2*pi*rand;

        EHj = Aj*cos(theta)*exp(1j*phi);

        EVj = Aj*sin(theta)*exp(1j*phi);

        Qe = zeros(1,Nphi);
        Qo = zeros(1,Nphi);

        for k = 1:Nphi

            EH = alpha*sin(beta)*exp(1j*dphi(k));

            EV = 1 + alpha*cos(beta)*exp(1j*dphi(k));

            Qe(k) = abs(EV)^2/(abs(EH)^2 + eps0);

            EHobs = EH + EHj ...
                + sigma*(randn+1j*randn);

            EVobs = EV + EVj ...
                + sigma*(randn+1j*randn);

            Qo(k) = abs(EVobs)^2/(abs(EHobs)^2 + eps0);

        end

        ue = abs(10*log10(Qo) - 10*log10(Qe));

        ubar = mean(ue);

        if ubar > gamma

            detections = detections + 1;

        end

    end

    PDjsr(m) = detections/MCjsr;

    fprintf('JSR = %+3d dB --> PD = %.4f\n', ...
        JSR_dB, PDjsr(m));

end

%% ==========================================================
% PD VS JSR PLOT
%% ==========================================================%% ==========================================================
% PD VS JSR
%% ==========================================================

fprintf('\nRunning matched-adversary JSR sweep...\n');

JSRlist = -20:2:20;

PDjsr = zeros(size(JSRlist));

for m = 1:length(JSRlist)

    JSR_dB = JSRlist(m);

    detections = 0;

    MCjsr = 1000;

    k = 10^(JSR_dB/20);

    for mc = 1:MCjsr

        Qe = zeros(1,Nphi);
        Qo = zeros(1,Nphi);

        for idx = 1:Nphi

            EH = alpha*sin(beta)*exp(1j*dphi(idx));

            EV = 1 + alpha*cos(beta)*exp(1j*dphi(idx));

            Qe(idx) = abs(EV)^2/(abs(EH)^2 + eps0);

            %% matched adversary

            EHj = k*EH;

            EVj = k*EV;

            EHobs = EH + EHj ...
                + sigma*(randn+1j*randn);

            EVobs = EV + EVj ...
                + sigma*(randn+1j*randn);

            Qo(idx) = abs(EVobs)^2/(abs(EHobs)^2 + eps0);

        end

        ue = abs(10*log10(Qo) - 10*log10(Qe));

        ubar = mean(ue);

        if ubar > gamma

            detections = detections + 1;

        end

    end

    PDjsr(m) = detections/MCjsr;

    fprintf('JSR = %+3d dB --> PD = %.4f\n', ...
        JSR_dB, PDjsr(m));

end

%% ==========================================================
% SUMMARY
%% ==========================================================

fprintf('\n');
fprintf('=====================================\n');
fprintf('PHASE-POLARIZATION DETECTOR SUMMARY\n');
fprintf('=====================================\n');
fprintf('Mean H0 statistic = %.4f\n',mean(u_H0));
fprintf('Mean H1 statistic = %.4f\n',mean(u_H1));
fprintf('Threshold         = %.4f\n',gamma);
fprintf('PD @ 0 dB JSR     = %.4f\n',PD);
fprintf('PFA               = %.4f\n',PFA);
fprintf('=====================================\n');