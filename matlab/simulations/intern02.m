clc;
clear;
close all;
A1 = 1;                  
A2 = 2;                  
fc = 1;                 
phi = pi/4;
sigma = 1;               
MC = 100000;             
t0 = 0.2;
s = A1*cos(2*pi*fc*t0);
j = A2*sin(2*pi*fc*t0 + phi);
T_H0 = zeros(1,MC);
T_H1 = zeros(1,MC);
for k = 1:MC
    w0 = sigma*randn;
    w1 = sigma*randn;
    y0 = s + w0;
    y1 = s + j + w1;
    T_H0(k) = j*(y0 - s);
    T_H1(k) = j*(y1 - s);
end
thresholds = linspace(min(T_H0),max(T_H1),500);
Pd = zeros(size(thresholds));
Pfa = zeros(size(thresholds));
for k = 1:length(thresholds)
    lambda = thresholds(k);
    %% Probability of detection
    Pd(k) = mean(T_H1 > lambda);
    %% Probability of false alarm
    Pfa(k) = mean(T_H0 > lambda);
end
%% PLOT ROC
figure;
plot(Pfa,Pd,'LineWidth',2);
xlabel('Probability of False Alarm');

ylabel('Probability of Detection');

title('ROC Curve for Instantaneous Bayesian LRT');

grid on;

%% THEORETICAL THRESHOLD

gamma = (j^2)/2;

disp('Theoretical Bayesian Threshold = ');
disp(gamma);