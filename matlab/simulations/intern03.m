clc;
clear;
close all;

%% PARAMETERS

N = 100;              % Number of samples

s = ones(1,N);        % Legitimate signal

sigma = 0.3;          % Noise std

%% JAMMER PATTERN

j = 2*sin(2*pi*(1:N)/20);

%% ---------------------------------------
%% CASE 1 : NO JAMMER
%% ---------------------------------------

n1 = sigma*randn(1,N);

y1 = s + n1;

%% Receiver estimate of signal
% (for now assume ideal estimate)

s_hat1 = s;

%% Residual

e1 = y1 - s_hat1;

%% ---------------------------------------
%% CASE 2 : JAMMER PRESENT
%% ---------------------------------------

n2 = sigma*randn(1,N);

y2 = s + j + n2;

%% Estimated signal

s_hat2 = s;

%% Residual

e2 = y2 - s_hat2;

%% ---------------------------------------
%% ENERGY COMPARISON
%% ---------------------------------------

E1 = sum(e1.^2);

E2 = sum(e2.^2);

disp(['Residual energy without jammer = ',num2str(E1)])

disp(['Residual energy with jammer    = ',num2str(E2)])

%% ---------------------------------------
%% PLOTS
%% ---------------------------------------

figure;

%% Case 1

subplot(2,2,1)

plot(y1,'LineWidth',1.5)
hold on
plot(s,'--','LineWidth',1.5)

title('No Jammer : Received vs Signal')

legend('Received y','Signal s')

grid on

%% Residual 1

subplot(2,2,2)

plot(e1,'LineWidth',1.5)

title(['Residual (No Jammer), Energy = ',num2str(E1)])

grid on

%% Case 2

subplot(2,2,3)

plot(y2,'LineWidth',1.5)
hold on
plot(s,'--','LineWidth',1.5)

title('Jammer Present : Received vs Signal')

legend('Received y','Signal s')

grid on

%% Residual 2

subplot(2,2,4)

plot(e2,'LineWidth',1.5)

title(['Residual (Jammer), Energy = ',num2str(E2)])

grid on