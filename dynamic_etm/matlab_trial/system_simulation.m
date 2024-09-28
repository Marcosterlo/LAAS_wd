function [x, u] = system_simulation(sysP, sysN, x0, nstep)

% System variables unpacking
A = sysP.A;
B = sysP.B;

% Neural network controller unpacking
n = sysN.n; % vector holding number of neurons per layer
nlayer = sysN.nlayer;
W = sysN.W; % weights unpacking
b = sysN.b; % biases unpacking

% Empty vectors initialization
% System state and input
x = zeros(sysP.nx, nstep);
u = zeros(sysP.nu, nstep);

% NN vectors
w = cell(sysN.nlayer, 1);
v = cell(sysN.nlayer, 1);
for i = 1:nlayer
    w{i,1} = zeros(n(i),1);
    v{i,1} = zeros(n(i),1);
end

% Initial conditions
x(:, 1) = x0;
x(:, 2) = x0;
u(1) = 0;

% Maximum bound in control input, used in saturation function
vbound = 3;

% Main loop for each iteration, starting from step 2 ignoring initial
% conditions
for k = 2:nstep
    
    % Neural network computations

    % Input layer
    v{1, k} = W{1}*x(:,k) + b{1};
    w{1, k} = sign(v{1, k}).*min(abs(v{1, k}), vbound);
    % Other layers
    for i = 2:nlayer
        v{i, k} = W{i}*w{i-1, k} + b{i};
        w{i, k} = sign(v{i, k}).*min(abs(v{i, k}), vbound);
    end

    % Input extraction from last layer
    u(:, k) = W{end}*w{end, k} + b{end};
    
    % State computation
    x(:, k+1) = A*x(:, k) + B*u(:, k);

end

end