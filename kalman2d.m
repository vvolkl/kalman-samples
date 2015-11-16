% -*- comment-start: "%"; -*-

clear;

global g dt;

N =  20;
dt = 0.2;
g = 9.81;

%%%% FIRST START WITH CREATING TRUE TRAJECTORY AND MEASURED (SMEARED) ONE
y0true = [ 100; 0]
ytrue = zeros(2,N);
ymeas = zeros(2,N);
sigma = 2;
for i=1:N
         ytrue(1, i) = y0true(1) - (i-1)*g*dt - (i-1)*g*dt^2/2;
         ytrue(2, i) = y0true(2) - (i-1)*g*dt;
         ymeas(:,i) = normrnd(ytrue(:,i),sigma);
end

%%%% CREATE INITIAL
%state vector
x = zeros(2,N);
x(:,1) = ymeas(:,1)*0.5 ;
P = [10*sigma 0;
        0 10*sigma];
V = [1 0;
       0 1];
R = [sigma^2 0
       0 sigma^2];
A = [1 dt;
       0 1];
H = [1 0;
       0 1];

%%%% KALMAN
x_= zeros(2,N);
for t=2:N
    x_(:,t) =[ x(1,t-1) - g*dt - g*dt^2/2
               x(2,t-1) - g*dt ];

    P_ = A*P*A';
    K = P_*H'/(H*P_*H'+V*R*V');
    x(:,t) = x_(:,t) + K*(ymeas(:,t)-x_(:,t));
    P =(eye(2)-K*H)*P_;
end

figure;
subplot(2,1,1);
plot(ytrue(1,:), 'b',ymeas(1,:), 'rx', x_(1,:), 'g+', x(1,:), 'k+');
xlabel("Iteration");
ylabel("Position");
subplot(2,1,2);
plot(ytrue(2,:), 'b',ymeas(2,:), 'rx', x_(2,:), 'g+', x(2,:), 'k+');
xlabel("Iteration");
ylabel("Velocity");
print -dpng kalman1D_stateVector2D.png

