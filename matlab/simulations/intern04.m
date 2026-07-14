clc;
clear;
close all;

%% USER INPUTS

alpha = input('Enter attenuation factor alpha = ');
beta_deg = input('Enter polarization rotation beta (deg) = ');

beta = deg2rad(beta_deg);

%% Phase sweep

dphi_deg = linspace(0,360,2000);
dphi = deg2rad(dphi_deg);

%% Arrays

psi_deg = zeros(size(dphi));

%% Compute polarization angle

for k = 1:length(dphi)

    % Horizontal component
    EH = alpha*sin(beta)*exp(1j*dphi(k));

    % Vertical component
    EV = 1 + alpha*cos(beta)*exp(1j*dphi(k));

    % Polarization angle
    psi = atan2(abs(EV),abs(EH));

    psi_deg(k) = rad2deg(psi);

end

%% Reference polarization
psi0 = 90;

%% Polarization deviation

dpsi = abs(psi_deg - psi0);

%% Plot

figure;

plot(dphi_deg,dpsi,'LineWidth',2);

xlabel('\Delta\phi (deg)','FontSize',12);
ylabel('\Delta\psi (deg)','FontSize',12);

title(sprintf('Phase-Polarization Relation (\\alpha = %.2f, \\beta = %.1f^o)', ...
               alpha,beta_deg));

grid on;
xlim([0 360]);

%% Display peak information

[max_dev,idx] = max(dpsi);

fprintf('\nMaximum polarization deviation = %.2f deg\n',max_dev);
fprintf('Occurs at phase shift = %.2f deg\n',dphi_deg(idx));