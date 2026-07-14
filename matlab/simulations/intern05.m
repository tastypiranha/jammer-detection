clc;
clear;
close all;

%% MULTIPATH PARAMETERS

alpha = input('Enter attenuation alpha = ');
beta_deg = input('Enter polarization rotation beta (deg) = ');

beta = deg2rad(beta_deg);

%% JAMMER PARAMETERS

Aj = input('Enter jammer amplitude Aj = ');

theta_j_deg = input('Enter jammer polarization angle (deg) = ');

phi_j_deg = input('Enter jammer phase (deg) = ');

theta_j = deg2rad(theta_j_deg);
phi_j   = deg2rad(phi_j_deg);

%% Phase sweep

dphi_deg = linspace(0,360,2000);
dphi = deg2rad(dphi_deg);

%% Arrays

Q_exp = zeros(size(dphi));
Q_jam = zeros(size(dphi));

%% Loop

for n = 1:length(dphi)

    %% Multipath field

    EH = alpha*sin(beta)*exp(1j*dphi(n));

    EV = 1 + alpha*cos(beta)*exp(1j*dphi(n));

    %% Expected

    Q_exp(n) = abs(EV)^2 / abs(EH)^2;

    %% Jammer field

    EHj = Aj*cos(theta_j)*exp(1j*phi_j);

    EVj = Aj*sin(theta_j)*exp(1j*phi_j);

    %% Total field

    EH_tot = EH + EHj;

    EV_tot = EV + EVj;

    %% Jammed Q

    Q_jam(n) = abs(EV_tot)^2 / abs(EH_tot)^2;

end

%% dB

QdB_exp = 10*log10(Q_exp);

QdB_jam = 10*log10(Q_jam);

%% --------------------------------------------------
%% Plot 1 : Linear
%% --------------------------------------------------

figure;

plot(dphi_deg,Q_exp,'LineWidth',2);
hold on;

plot(dphi_deg,Q_jam,'LineWidth',2);

xlabel('\Delta\phi (deg)');
ylabel('Q = |E_V|^2 / |E_H|^2');

title('Multipath Manifold vs Jammed Manifold');

legend('Expected','With Jammer');

grid on;

%% --------------------------------------------------
%% Plot 2 : dB
%% --------------------------------------------------

figure;

plot(dphi_deg,QdB_exp,'LineWidth',2);
hold on;

plot(dphi_deg,QdB_jam,'LineWidth',2);

xlabel('\Delta\phi (deg)');
ylabel('Q_{dB} (dB)');

title('Q_{dB} : Multipath vs Jammed');

legend('Expected','With Jammer');

grid on;

%% --------------------------------------------------
%% Deviation Statistic
%% --------------------------------------------------

u = abs(QdB_jam - QdB_exp);

figure;

plot(dphi_deg,u,'LineWidth',2);

xlabel('\Delta\phi (deg)');
ylabel('u (dB)');

title('Phase-Polarization Consistency Error');

grid on;

fprintf('\nMaximum deviation u = %.2f dB\n',max(u));
fprintf('Mean deviation u = %.2f dB\n',mean(u));