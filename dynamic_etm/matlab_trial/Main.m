clearvars
clc

%% Parameters
% g = 9.81; % grav coeff
g = 10; % grav coeff
m = 0.15; % mass
l = 0.5; % length
mu = 0.05; % frict coeff
dt = 0.02; % sampling period

%% System definition: x^+1 = A*x + B*u

% Matrix definition
sysP.A = [1,                dt;
          g/l*dt,     1 - mu/(m*l^2)*dt];
sysP.B = [0;
          dt/(m*l^2)];

% Size of state x
sysP.nx = size(sysP.A, 1);

% Size of input u
sysP.nu = 1;

% I have to create 2 variables holding the last state used to update the
% input and its time instant
sysP.last_x = zeros(sysP.nx, 1);
sysP.last_t = 0;

%% Loading weights and biases of trained NN controller

load('weight_saturation.mat')
W{1} = W1;
W{2} = W2;
W{3} = W3;
b{1} = b1;
b{2} = b2;
b{3} = b3;

% Number of layers
nlayer = numel(W) - 1;
sysN.nlayer = nlayer;

% I create a system in an analogous way for the system representing the
% trained NN

% n is a vector storing the number of neurons per each layer
n = zeros(1, nlayer+1);
for i = 1:nlayer+1
    n(i) = size(W{i},1);
    sysN.W{i} = W{i};
    sysN.b{i} = b{i};
end
sysN.n = n;
% I embed every quantity in sysN in order to pass fewer arguments in the
% functions and unpack everything inside the functions

% Size of vector phi (# of activation functions)
sysN.nphi = sum(n(1:nlayer));


%% ETM paremeters estimation

% [psi, lambda, rho] = etm_parameters(sysP, sysN);


%% Simulation

% Initial state
x0 = [0.3, 0];

% Simulation steps
nstep = 300;

% Call to closed loop simulation
[x, u] = system_simulation(sysP, sysN, x0, nstep);

%% Plots

% Time vector creation
t = 0:nstep;

% Angle plot
figure(1)
plot(t, x(1, :))
xlabel("Time [s]")
ylabel("Angle [rad]")
title("Angle evolution")

% Velocity plot
figure(2)
plot(t, x(2, :))
xlabel("Time [s]")
ylabel("Velocity [rad/s]")
title("Angular velocity evolution")

% Control input plot
figure(3)
plot(t(2:end), u)
xlabel("Time [s]")
ylabel("Input")
title("Input evolution")