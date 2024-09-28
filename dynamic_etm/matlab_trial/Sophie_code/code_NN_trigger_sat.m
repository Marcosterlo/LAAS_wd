function [sol,solPb] = code_NN_trigger_sat(sysP,sysC,x1b,vb,mu)

pbLMI = [];
Acl = sysP.AG;
Bcl = sysP.BG;

nG = size(Acl,1);
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
Nux = N(nphi+1:end,1:nG);
Nuw = N(nphi+1:end,nG+1:end);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);

%%
% Definition des variableset du systeme de LMIs a contruire
% a = sdpvar(1,1);   
% pbLMI = pbLMI + (a>=0);
P = sdpvar(nG,nG,'symmetric');   
pbLMI = pbLMI + (P>=1e-08*eye(size(P,1)));
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);%diag(mu)
Z = sdpvar(nphi,nG,'full');
      
Rphis = [eye(nG) zeros(nG,nphi);
        Nvx Nvw;
        zeros(nphi,nG) eye(nphi)];     

Mphi = [zeros(nG) -Z' Z';
          -Z zeros(nphi) T;
           Z T -2*T];  
        
Qphis = Rphis'*Mphi*Rphis; 

Rs = [eye(nG) zeros(nG,nphi);
      Nux Nuw];

lmi11 = Acl'*P*Acl-P;
lmi12 = Acl'*P*Bcl;
lmi22 = Bcl'*P*Bcl;

Qs = [lmi11  lmi12;
      lmi12' lmi22];

Qlyap = Rs'*Qs*f;
  
MSTAB =  Qlyap  + Qphis; 

%D = blkdiag(eye(nG), eye(nphi));
pbLMI = pbLMI + (MSTAB <= -1e-10*eye(size(MSTAB,1)));% + (D'*MSTAB*D>=-a*eye(nG+nphi));

lmi111 = P; 
for i = 1:nphi
    lmi121 = Z(i,:);
    lmi122 = 2*mu(i)*T(i,i)-mu(i)^(2)*vb(i)^(-2);%mu(i)*T(i,i)*vb(i)^(2);%
    MSETOR = [lmi111 lmi121';
    lmi121 lmi122];
    pbLMI = pbLMI + (MSETOR>= 1e-10*eye(size(MSETOR,1)));%
end

lmi211 = x1b^2;
lmi212 = [1,0];
lmi222 = P;
MSETOR2 = [lmi211 lmi212;
    lmi212' lmi222];
pbLMI = pbLMI + (MSETOR2>= 1e-10*eye(size(MSETOR2,1)));

% critere d'optimisation
critOPTIM = trace(P);%trace(Qw(1:n1,1:n1));%(n1+1:n1+n2,n1+1:n1+n2)
% les options d'optimisation);%trace(Qw);%%%%
% les options d'optimisation
% --------------------------
options_sdp = sdpsettings('verbose',0,'warning',0,'solver','sdpt3');
options_sdp.sdpt3.maxit    = 200;
options_sdp.lmilab.maxiter = 500;    % default = 100
options_sdp.lmilab.reltol = 0.001;    % default = 0.01?? in lmilab 1e-03
options_sdp.lmilab.feasradius = 1e9; % R<0 signifie "no bound", default = 1e9

% resolution du probleme
solPb = solvesdp(pbLMI,critOPTIM,options_sdp);
%solPb = optimize(pbLMI,critOPTIM,options_sdp);

feasible = min(checkset(pbLMI));
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