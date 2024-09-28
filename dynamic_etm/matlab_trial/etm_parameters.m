function [sol_solver, solution] = etm_parameters(sysP, sysN, varargin)

% The aim of this function is the design of the dynamic ETM parameters
% while keeping the regional stability of the closed loop

% Matrix that will include all lmi conditions
pbLMI = [];

% System unpacking
A = sysP.A;
B = sysP.B;

% NN unpacking
% n = sysN.n; % vector holding number of neurons per layer
nlayer = sysN.nlayer;
nphi = sysN.nphi; % Size of vector phi (# of activation functions)
W = sysN.W;
b = sysN.b;

% Variables holding vector dimension
nx = size(A, 1); % dim of state x
% nu = size(B, 2); % dim of state y


% Creation of Matrix N
N = [];
for i = 1:nlayer+1
    N = blkdiag(N, W{i});
end

% Creation of submatrices of N
Nux = N(nphi+1:end,1:nx);
Nuw = N(nphi+1:end,nx+1:end);
Nvx = N(1:nphi,1:nx);
Nvw = N(1:nphi,nx+1:end);
Nub = b{3};
Nvb = [b{1}; b{2}];

%% Variables for LMI

% P matrix used in Lyapunov function
P = sdpvar(nx, nx, 'symmetric');
pbLMI = pbLMI + (P >= 1e-03*eye(size(P,1)));

if ~isempty(varargin)
    % Parameter rho used to determine the decay rate of dynamic ETM 
    % threshold
    rho = varargin{1}; % rho >= 0
    % Parameter lambda used in conjunction with rho to determine the 
    % decay rate of the dynamic paraemter eta
    lambda = varargin{2}; % 0 <= (lambda + rho) < 1
    % tau var used to apply assumption 1 for non linearities abstraction
    tau = varargin{3}; % tau >= 0
    fprintf('\nUsing passed parameters:\n');
else
    rho = 0.2; 
    lambda = 0.2;     
    tau = 0.1;
    fprintf("\nUsing default values for parameters\n");
end

% Printing parameters
disp("rho: " + rho);
disp("lambda: " + lambda);
disp("tau: " + tau);

% Matrix of parameters definition for quadratic function in the ETM
psi1 = sdpvar(nphi, nphi);
psi2 = sdpvar(nphi, nphi);
psi3 = sdpvar(nphi, nphi);
pbLMI = pbLMI + (psi1 + psi2 + psi2' + psi3 <= 0);

% Matrices S R T used for the quadratic abstraction
S = sdpvar(nphi, nphi, 'symmetric');
pbLMI = pbLMI + (S >= 0);
R = sdpvar(nphi, nphi, 'symmetric');
pbLMI = pbLMI + (R <= 0);
T = sdpvar(nphi, nphi);

%% Matrix M creation

% Non zero terms of M matrix
lmi11 = A'*P*A - P + A'*P*B*Nux + Nux'*B'*P*A + Nux'*B'*P*B*Nux ...
    + tau*Nvx'*S*Nvx;
lmi13 = A'*P*B*Nuw + Nux'*B'*P*B*Nuw + tau * (Nvx'*S*Nvw + Nvx'*T);
lmi22 = (lambda - 1)/rho * psi3;
lmi23 = (lambda - 1)/rho * psi2';
lmi33 = Nuw'*B'*P*B*Nuw + (lambda - 1)/rho * psi1 + tau * (R + ...
    Nvw'*S*Nvw + Nvw'*T + T'*Nvw);

% Null terms of M matrix
lmi12 = zeros(size(lmi11, 1), size(lmi22, 2));

M = [lmi11  lmi12  lmi13;
     lmi12' lmi22  lmi23;
     lmi13' lmi23' lmi33];

% Equilibrium conditions
% First create the matrices of the paper
R = eye(size(Nvw)) - inv(Nvw + 1e-3 * eye(size(Nvw)));
Rw = Nux + Nuw*R*Nvx;
Rb = Nuw*R*Nvb + Nub;

% Equilibrium values
xstar = (eye(size(A)) - A - B * Rw)\B*Rb;
vstar = R*Nvx*xstar + R*Nvb;
ustar = Rw*xstar + Rb;
chistar = vstar;
phistar = vstar;

pbLMI = pbLMI + (M <= -1e-3);

fprintf("\nList of constraints for the LMI problem\n")
pbLMI



%% LMI solution

critOPTIM = trace(P);

options_sdp = sdpsettings('verbose',0,'warning',0,'solver','sdpt3');
options_sdp.sdpt3.maxit    = 200;
options_sdp.lmilab.maxiter = 500;    
options_sdp.lmilab.reltol = 0.001;    
options_sdp.lmilab.feasradius = 1e9;

fprintf("\nInfo on resolution:\n")
sol_solver = solvesdp(pbLMI,critOPTIM,options_sdp)

% Outputs packing

solution.M = double(M);
solution.P = double(P);
solution.psi1 = double(psi1);
solution.psi2 = double(psi2);
solution.psi3 = double(psi3);
solution.mat = solution.psi1 + solution.psi2 + solution.psi2' ...
    + solution.psi3;
solution.S = double(S);
solution.T = double(T);
solution.R = double(R);

disp("Max eigenvalue of P: " + max(eig(solution.P)))
disp("Max eigenvalue of the sum of psi: " + max(eig(solution.mat)))
disp("Max eigenvalue of M: " + max(eig(solution.M)))

end

