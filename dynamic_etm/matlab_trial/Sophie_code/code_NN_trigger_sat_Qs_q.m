function [sol,solPb] = code_NN_trigger_sat_Qs_q(sysP,sysC,x1bound,deltav1,mu)

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
% % Definition des variables et du systeme de LMIs a contruire
a = sdpvar(1,1,'full');
pbLMI = pbLMI + (a>=0);
P = sdpvar(nzeta,nzeta,'symmetric');   
pbLMI = pbLMI + (P>=1e-8*eye(size(P,1)));
% dn1 = sdpvar(n(1),n(1),'symmetric');
% pbLMI = pbLMI + (dn1>=0);
% dn2 = sdpvar(n(2),n(2),'symmetric');
% pbLMI = pbLMI + (dn2>=0);
% bn1 = sdpvar(nG,nG,'symmetric');
% pbLMI = pbLMI + (bn1>=0);
% vn1 = sdpvar(nG,nG,'symmetric');
% pbLMI = pbLMI + (vn1>=0);
% bn2 = sdpvar(n(1),n(1),'symmetric');
% pbLMI = pbLMI + (bn2>=0);
% Qw = blkdiag(dn1+bn2,dn2);
Qe = [];
Qw = [];
for i=1:nlayer
    Qe = blkdiag(Qe,sdpvar(n(i),n(i),'diagonal'));
    Qw = blkdiag(Qw,sdpvar(n(i),n(i),'diagonal'));
end
pbLMI = pbLMI + (Qw>=0);%+ (Qw>=b*eye(nphi));
% rn1 = sdpvar(n(1),n(1),'symmetric');
% pbLMI = pbLMI + (rn1>=0);
% rn2 = sdpvar(n(2),n(2),'symmetric');
% pbLMI = pbLMI + (rn2>=0);
% Qe = blkdiag(rn1-bn2,rn2); 
pbLMI = pbLMI + (Qe>=0);%+ (Qe<=eye(nphi));
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);
Z = sdpvar(nphi,nG,'full');
lambda1 = sdpvar(1,1);
pbLMI = pbLMI + (lambda1>=0);
lambda2 = sdpvar(1,1);
pbLMI = pbLMI + (lambda2>=0);

Rphis = [eye(nzeta) zeros(nzeta,nphi) zeros(nzeta,nphi) zeros(nzeta,nq);
        Nvzeta Nvw zeros(nphi,nphi) zeros(nphi,nq);
        zeros(nphi,nzeta) eye(nphi) zeros(nphi,nphi) zeros(nphi,nq)];      

Zxi = [Z zeros(nphi,nxi)];  
Mphi = [zeros(nzeta) Zxi' -Zxi';
           Zxi -2*T 2*T;
          -Zxi 2*T -2*T];  
    
Qphis = Rphis'*Mphi*Rphis;   
  
lmi11 = (Acl+Bcl*Nuzeta)'*P*(Acl+Bcl*Nuzeta)-P;%+ [bn1+vn1 zeros(nG,nxi);zeros(nxi,nG+nxi)]; 
lmi12 = (Acl+Bcl*Nuzeta)'*P*Bcl*Nuw;
lmi13 = (Acl+Bcl*Nuzeta)'*P*Bcl*Nuw;
lmi14 = (Acl+Bcl*Nuzeta)'*P*Bq;
lmi22 = Nuw'*Bcl'*P*Bcl*Nuw + Qw;
lmi23 = Nuw'*Bcl'*P*Bcl*Nuw;%+ [bn2 zeros(n(1),n(2));zeros(n(2),n(1)+n(2))];
lmi24 = Nuw'*Bcl'*P*Bq;
lmi33 = Nuw'*Bcl'*P*Bcl*Nuw -Qe;
lmi34 = Nuw'*Bcl'*P*Bq;
lmi44 = Bq'*P*Bq;

QV = [lmi11  lmi12  lmi13 lmi14;
      lmi12' lmi22  lmi23 lmi24;
      lmi13' lmi23' lmi33 lmi34;
      lmi14' lmi24' lmi34' lmi44];  
     
Qoff = lambda2*[Ccl+Dcl*Nuzeta Dcl*Nuw Dcl*Nuw Dq]'...
    *M_off*[Ccl+Dcl*Nuzeta Dcl*Nuw Dcl*Nuw Dq];        

R_sec = [eye(1,1),zeros(1,1+nxi+nphi+nphi+nq);...
       zeros(nq,nzeta+nphi+nphi),eye(nq)];
   
Qsec = lambda1*R_sec'*Psi_sec'*M_sec*Psi_sec*R_sec;
    
MSTAB = Qoff + QV + Qsec + Qphis;

D = blkdiag(eye(nzeta), zeros(2*nphi), zeros(nq));
pbLMI = pbLMI + (MSTAB <= 0) + (D'*MSTAB*D>=-a*eye(nzeta+2*nphi+nq));

lmi111 = P; 
for i = 1:n(1)
    lmi121 = [T(i,i)*W{1}(i,:)-Z(i,:) zeros(1,nxi)];
    lmi122 = 2*mu*T(i,i)-mu^(2)*deltav1^(-2);%alpha*T(i,i)*deltav1^(2);%
    MSETOR = [lmi111 lmi121';
    lmi121 lmi122];
    pbLMI = pbLMI + (MSETOR>=0);
end
% 
lmi211 = x1bound^2;
lmi212 = [1,0,zeros(1,nxi)];
lmi222 = P;
MSETOR2 = [lmi211 lmi212;
    lmi212' lmi222];
pbLMI = pbLMI + (MSETOR2>=0);

% critere d'optimisation
critOPTIM = a;% - trace(bn1)- trace(vn1); %
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

feasible = min(checkset(pbLMI));
if (feasible >= 0) % && solPb.problem == 0)
    sol.P = double(P);
    sol.Qe = double(Qe);%double(blkdiag(rn1,rn2));%
    sol.Qw = double(Qw);%double(blkdiag(bn1,dn1,bn2,dn2));%
    sol.T = double(T);
    sol.G = sol.T\double(Z);
else
    sol.P = [];
    sol.Qe = [];
    sol.Qw = [];
    sol.G = [];
end

end