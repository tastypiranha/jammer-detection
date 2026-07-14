clc;
clear;
close all;
A1 = 1;
A2 = 2;
fc = 1;
phi = pi/4;
t = linspace(0,2,2000);
s = A1*cos(2*pi*fc*t);
g = sin(2*pi*fc*t + phi);
gamma = s + (A2/2)*g;
idx_pos = g > 0;
idx_neg = g < 0;
%% =====================================================
%% CASE 1 : sin(.) > 0
%% =====================================================
figure;
subplot(2,1,1);
plot(t(idx_pos),gamma(idx_pos), ...
    'y','LineWidth',3);
xlabel('Time');
ylabel('\gamma(t)');
title('CASE 1 : sin(2\pi f_c t + \phi) > 0');
legend('\gamma(t)');
grid on;
%% =====================================================
%% CASE 2 : sin(.) < 0
%% =====================================================
subplot(2,1,2);
plot(t(idx_neg),gamma(idx_neg), ...
    'y','LineWidth',3);
xlabel('Time');
ylabel('\gamma(t)');
title('CASE 2 : sin(2\pi f_c t + \phi) < 0');
legend('\gamma(t)');
grid on;