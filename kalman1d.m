% -*- comment-start: "%"; -*-

clear;

global g dt;

N =  20;
dt = 0.2;
g = 9.81;

%%%% FIRST START WITH CREATING TRUE TRAJECTORY AND MEASURED (SMEARED) ONE
y0true = 100;
ytrue = zeros(1,N);
ymeas = zeros(1,N);
sigma = 2;
for i=1:N
    ytrue(i) = y0true - (i-1)*g*dt - (i-1)*g*dt^2/2;
    ymeas(i) = normrnd(ytrue(i),sigma);
end

%%%% CREATE INITIAL
%state vector
x = zeros(1,N);
x(1) = ymeas(1)*0.5 ;

% covariance
P = 10;

%measurement variance
R = 1;

%%%% KALMAN
x_= zeros(1,N);
for t=2:N
    x_(t) =x(t-1) - g*dt - g*dt^2/2;
    P_ = P;
    K = P_/(P_+R);
    x(t) = x_(t) + K*(ymeas(t)-x_(t));
    P =(1-K)*P_;
end

figure;
plot(ytrue, 'b',ymeas, 'rx', x_, 'g+', x, 'k+');
xlabel('Iteration');
ylabel('Position');
legend('True trajectory', 'Measured position', 'Prediction','Kalman');
print -dpng kalman1d.png

