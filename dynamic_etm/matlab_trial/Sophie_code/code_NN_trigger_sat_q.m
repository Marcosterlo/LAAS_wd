function [sol,solPb] = code_NN_trigger_sat_q(sysP,sysC,x1bound,deltav1,mu)

pbLMI = [];
Acl = sysP.A;
Bcl = sysP.Bu;
Bq = sysP.Bq;
Ccl = sysP.C;
Dcl = sysP.Du;
Dq = sysP.Dq;

nG = sysP.nG;
nxi = sysP.nxi;
nzeta = sysP.nzeta;
nu = 1;
nq = 1;
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
Nuzeta = [Nux, zeros(nu,nxi)];
Nvzeta = [Nvx, zeros(nphi,nxi)];

% M matrix for off-by-one IQC 
M_off = [0, 1;
         1, 0];
% define the filter for sector IQC
%L_sector = 0.7606;
L_sector = (x1bound - sin(x1bound))/x1bound;
m_sector = 0;
Psi_sec = [L_sector, -1;...
           -m_sector,    1];
% M matrix for sector IQC
M_sec = [0, 1;...
         1, 0];

%%
% Definition des variableset du systeme de LMIs a contruire
a = sdpvar(1,1);   
pbLMI = pbLMI + (a>=0);
P = sdpvar(nzeta,nzeta,'symmetric');   
pbLMI = pbLMI + (P>=0);
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);%
Z = sdpvar(nphi,nG,'full');
lambda1 = sdpvar(1,1);
pbLMI = pbLMI + (lambda1>=0);
lambda2 = sdpvar(1,1);
pbLMI = pbLMI + (lambda2>=0);
      
Rphis = [eye(nzeta) zeros(nzeta,nphi) zeros(nzeta,nq);
        Nvzeta Nvw zeros(nphi,nq);
        zeros(nphi,nzeta) eye(nphi) zeros(nphi,nq)];     

Zxi = [Z zeros(nphi,nxi)];  
Mphi = [zeros(nzeta) Nvzeta'*T-Zxi' -(Nvzeta'*T-Zxi');
           T*Nvzeta-Zxi -2*T 2*T+T*Nvw;
          -(T*Nvzeta-Zxi) 2*T+Nvw'*T -T*(eye(nphi)+Nvw)-(eye(nphi)+Nvw)'*T];  
     
Qphis = Rphis'*Mphi*Rphis; 

Rs = [eye(nzeta) zeros(nzeta,nphi) zeros(nzeta,nq);
      Nuzeta Nuw zeros(nu,nq);
      zeros(nq,nzeta) zeros(nq,nphi) eye(nq)];

lmi11 = Acl'*P*Acl-P;
lmi12 = Acl'*P*Bcl;
lmi13 = Acl'*P*Bq;
lmi22 = Bcl'*P*Bcl;
lmi23 = Bcl'*P*Bq;
lmi33 = Bq'*P*Bq;

Qs = [lmi11  lmi12 lmi13;
      lmi12' lmi22 lmi23;
      lmi13' lmi23' lmi33];

Qlyap = Rs'*Qs*Rs;
  
Qoff = lambda2*Rs'*[Ccl Dcl Dq]'...
    *M_off*[Ccl Dcl Dq]*Rs;     

R_sec = [eye(1,1),zeros(1,1+nxi+nphi+nq);...
       zeros(nq,nzeta+nphi),eye(nq)];
   
Qsec = lambda1*R_sec'*Psi_sec'*M_sec*Psi_sec*R_sec;   

MSTAB =  Qlyap  + Qphis + Qoff + Qsec; 

D = blkdiag(eye(nzeta), eye(nphi), eye(nq));
pbLMI = pbLMI + (MSTAB <= 0) ;%+ (D'*MSTAB*D>=-a*eye(nzeta+nphi+nq));

lmi111 = P; 
for i = 1:nphi
    lmi121 = [Z(i,:) zeros(1,nxi)];
    lmi122 = 2*mu(i)*T(i,i)-mu(i)^(2)*deltav1(i)^(-2);%alpha*T(i,i)*deltav1^(2);%
    MSETOR = [lmi111 lmi121';
    lmi121 lmi122];
    pbLMI = pbLMI + (MSETOR>=0);
end

lmi211 = x1bound^2;
lmi212 = [1,0, zeros(1,nxi)];
lmi222 = P;
MSETOR2 = [lmi211 lmi212;
    lmi212' lmi222];
pbLMI = pbLMI + (MSETOR2>=0);

% critere d'optimisation
critOPTIM = trace(P(1:2,1:2));%trace(Qw(1:n1,1:n1));%(n1+1:n1+n2,n1+1:n1+n2)
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