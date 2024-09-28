function [sol,solPb] = code_NN_trigger_sat_Ch(sysP,sysC,x1b,vb,mu)

pbLMI = [];

AG = sysP.AG;
BG = sysP.BG;
nG = size(AG,1);
nlayer = numel(sysC.W)-1;

N = [];
n = zeros(1,nlayer);
W = cell(1,nlayer);
for i=1:nlayer+1
    W{i} = sysC.W{i};
    n(i) = size(W{i},1);
    N = blkdiag(N,W{i});
end
nphi = sum(n(1:nlayer));
K = [-0.1 0];
Nuw = N(nphi+1:end,nG+1:end);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);
%%
% Definition des variableset du systeme de LMIs a contruire
b = sdpvar(1,1,'full');
pbLMI = pbLMI + (b>=0);
P = sdpvar(nG,nG,'symmetric');   
pbLMI = pbLMI + (P>=1e-08*eye(size(P,1)));
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);%diag(mu)
Z = sdpvar(nphi,nG,'full');
R = (eye(nphi)+Nvw);%(eye(nphi)+Nvw);

Rphis = [eye(nG) zeros(nG,nphi);
        R*Nvx eye(nphi)-R;
        zeros(nphi,nG) eye(nphi)]; 

Mphi = [zeros(nG) zeros(nG,nphi) zeros(nG,nphi);
          Z -T T];

Qphi = Rphis'*Mphi'+ Mphi*Rphis;
      
AB = [AG+BG*K+BG*Nuw*R*Nvx -BG*Nuw*R];

MSTAB = AB'*P*AB - blkdiag(P, zeros(nphi)) - Qphi;

% lmi11 = (AG+BG*K+BG*Nuw*R*Nvx)'*P*(AG+BG*K+BG*Nuw*R*Nvx)-P; 
% lmi12 = -(AG+BG*K+BG*Nuw*R*Nvx)'*P*BG*Nuw*R-Z'+Nvx'*R'*T;
% lmi22 = R'*Nuw'*BG'*P*BG*Nuw*R-R'*T-T*R;
% 
% MSTAB = [lmi11  lmi12;
%       lmi12' lmi22];
  
D = blkdiag(eye(nG),eye(nphi));

pbLMI = pbLMI + (MSTAB <= 0) + (MSTAB >= -b*eye(size(MSTAB,1)));
   
lmi111 = P; 
for i = 1:nphi
    lmi121 = Z(i,:);
    lmi122 =  2*mu(i)*T(i,i)-mu(i)^(2)*vb(i)^(-2);
    MSETOR = [lmi111 lmi121';
    lmi121 lmi122];
    pbLMI = pbLMI + (MSETOR>=0);
end

lmi211 = x1b^2;
lmi212 = [1,0];
lmi222 = P;
MSETOR2 = [lmi211 lmi212;
    lmi212' lmi222];
pbLMI = pbLMI + (MSETOR2>=0);
% 
P0 = sdpvar(nG, nG,'symmetric');   
pbLMI = pbLMI + (P0>=0);

% P0 = 2*[0.0537    0.0061;
%     0.0061    0.0046];
P0 = 2*[0.1458    0.0027;
    0.0027    0.0045];
MSETOR3 = [P0 P;
    P P];
pbLMI = pbLMI + (MSETOR3>=0);

% critere d'optimisation
critOPTIM = b;%trace(Qw(1:n1,1:n1));%(n1+1:n1+n2,n1+1:n1+n2)
% les options d'optimisation);%trace(Qw);%%%%
% les options d'optimisation
% --------------------------
options_sdp = sdpsettings('verbose',0,'warning',0,'solver','sdpt3');
options_sdp.sdpt3.maxit    = 100;
options_sdp.lmilab.maxiter = 500;    % default = 100
options_sdp.lmilab.reltol = 0.001;    % default = 0.01?? in lmilab 1e-03
options_sdp.lmilab.feasradius = 1e9; % R<0 signifie "no bound", default = 1e9

% resolution du probleme
solPb = solvesdp(pbLMI,critOPTIM,options_sdp);
%solPb = optimize(pbLMI,critOPTIM,options_sdp);

feasible = min(checkset(pbLMI))
if (feasible >= 0) % && solPb.problem == 0)
    sol.P = double(P);
    sol.T = double(T);
    sol.G = sol.T\double(Z);
else
    sol.P = [];
    sol.T = [];
    sol.G = [];
end

end