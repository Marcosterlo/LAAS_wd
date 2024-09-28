function [x, u, eta] = dynamic_system_simulation(sysP, sysN, x0, rho, ...
    lambda, psi, bound, nstep)

close all

% System variables unpacking
A = sysP.A;
B = sysP.B;

% Neural network contorller unpacking
n = sysN.n; % vector holding number of neurons per layer
nlayer = sysN.nlayer;
W = sysN.W; % Weights unpacking
b = sysN.b; % biases unpacking

% Empty vectors initiailzation
% System state and input
x = zeros(sysP.nx, nstep);
u = zeros(sysP.nu, nstep);

% Vector holding eta values and instant of update values
eta = zeros(sysN.nlayer, nstep);
s = zeros(sysN.nlayer, nstep);
fival = zeros(1, nstep);

% NN vectors
% The use of cell is justified by the need to have 1 vector per layer and
% the number of layer must be initialized dynamically
w = cell(sysN.nlayer, 1);
v = cell(sysN.nlayer, 1);
% Last state used for update
chi = cell(sysN.nlayer, 1);
for i = 1:nlayer
    w{i, 1} = zeros(n(i), 1);
    v{i, 1} = zeros(n(i), 1);
    chi{i, 1} = zeros(n(i), 1);
end

% Initial conditions
x(:, 1) = x0;
u(1) = 0;
eta(:, 1) = 10;
s(:, 1) = 1;

% Maximum bound for activation functions, saturation
% bound = 5;

% Main loop, executed once per step
for k = 2:nstep
    
    % Dynamic ETM
    
    % not called at first step
    if k == 2
        % First layer

        v{1} = W{1}*x(:, k-1) + b{1};
        w{1} = sign(v{1}).*min(abs(v{1}), bound);
        % Since it's the first layer we propagate the current NN state and
        % hence we fix every chi value as the current w
        chi{1} = w{1};
        % Dynamic of eta not yet started for every layer
        eta(:, k) = eta(1, 1);
        % Marking first step as an update one for every layer
        s(:, k) = 1;
        % Other layers
        for i = 2:nlayer
            v{i} = W{i}*chi{i-1} + b{i};
            w{i} = sign(v{i}).*min(abs(v{i}), bound);
            chi{i} = w{i};
        end
    end

    % From the second step on

    % Flag variable to check if the update happened, if negative all
    % following steps are skipped
    check = 1;
    
    % First layer
    v{1} = W{1}*x(:, k-1) + b{1};
    w{1} = sign(v{1}).*min(abs(v{1}), bound);

    % Creation of [w; chi] vector
    vec = [w{1}; w{2}; chi{1}; chi{2}];  
    
    % Eta dynamics
    eta(1, k) = (lambda + rho)*eta(1, k-1) - vec'*psi*vec;

    % Dynamic ETM check
    if vec'*psi*vec <= rho*eta(1, k)

        % If check is positive the last state is updated
        chi{1} = w{1}; 
        % Step k is marked as an update one for first layer
        s(1, k) = 1;

    else

        % State not propagated, no need to compute other layers. Flag
        % variable set to 0
        check = 0;
        % Step k is marked as an non-update one for first layer
        s(1, k) = 0;

    end

    % All other layers
    for i = 2:nlayer
        if check
            % The output of the previous layer is chi
            v{i} = W{i}*chi{i-1} + b{i};
            w{i} = sign(v{i}).*min(abs(v{i}), bound);

            % Creation of [w; chi] vector
            vec = [w{1}; w{2}; chi{1}; chi{2}];  

            % Eta dynamics
            eta(i, k) = (lambda + rho)*eta(i, k-1) - vec'*psi*vec;
            fival(k) = vec'*psi*vec;

            % Same logic as before
            if vec'*psi*vec <= rho*eta(i, k)
                chi{i} = w{i}; 
                s(i, k) = 1;
            else
                check = 0;
                s(i, k) = 0;
            end
        end
    end
    
    % Only at first step the output is computed with the fully propagated
    % input
    if k==2
        u(k) = W{end}*w{end} + b{end};
        u(k) = sign(u(k)).*min(abs(u(k)), bound);
    else
        if check
            % If we reach the final layer with the variable check still set
            % to 1 it means that no ETM has been activated, hence the input
            % u is computed with the fully propagated NN
            u(k) = W{end}*w{end} + b{end};
            u(k) = sign(u(k)).*min(abs(u(k)), bound);
        else
            % If a ETM has been activated it means that from the last layer
            % point of view the state remains the same as before, hence we
            % can use the last used input
            u(k) = u(k-1);
        end
    end

    % Dynamics computation
    x(:, k+1) = A*x(:, k) + B*u(k);
end

%% Plots
% First layer eta
figure(1)
title("Debug")
subplot(2,2,1)
plot(1:nstep+1, x(1, :))
title("Position [rad]")
subplot(2,2,2)
plot(1:nstep+1, x(2, :))
title("Velocity [rad/s]")
subplot(2,2,3)
plot(1:nstep, eta(1, :))
title("First layer eta")
subplot(2,2,4)
stem(1:nstep, s(1, :))
ylim([-0.2, 1.5])
title("Update instants of 1st layer")

end