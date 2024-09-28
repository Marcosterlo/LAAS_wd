function [sol,solPb] = code_NN_trigger_sat_Qs_2(sysP,sysC,x1bound,deltav1,mu)

pbLMI = [];

AG = sysP.AG;
BG = sysP.BG2;
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
Nux = N(nphi+1:end,1:nG);
Nuw = N(nphi+1:end,nG+1:end);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);

%%
% Definition des variableset du systeme de LMIs a contruire
a = sdpvar(1,1,'full');
b = sdpvar(1,1,'full');
pbLMI = pbLMI + (a>=0) + (b>=0);
P = sdpvar(nG,nG,'symmetric');   
pbLMI = pbLMI + (P>=0);
%+ (P<= [0.4633 0.0258; 0.0258 0.0911]);
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);%diag(mu)
Z = sdpvar(nphi,nG,'full');
% Qe = [];
% Qw = [];
% for i=1:nlayer
%     Qe = blkdiag(Qe,sdpvar(n(i),n(i),'symmetric'));
%     Qw = blkdiag(Qw,sdpvar(n(i),n(i),'symmetric'));
% end
% pbLMI = pbLMI + (Qe>=0) + (Qe(1:n(1)+n(2),1:n(1)+n(2))<=10*eye(n(1)+n(2)));
% pbLMI = pbLMI + (Qw>=0) + (Qw(1+n(1):sum(n(1:2)),1+n(1):sum(n(1:2)))>=b*eye(n(2)));

dn1 = sdpvar(n(1),n(1),'symmetric');
pbLMI = pbLMI + (dn1>=0);
dn2 = sdpvar(n(2),n(2),'symmetric');
pbLMI = pbLMI + (dn2>=0);
dn3 = sdpvar(n(3),n(3),'symmetric');
pbLMI = pbLMI + (dn3>=0);
bn1 = sdpvar(nG,nG,'symmetric');
pbLMI = pbLMI + (bn1>=0);
bn2 = sdpvar(n(1),n(1),'symmetric');
pbLMI = pbLMI + (bn2>=0);
% bn3 = sdpvar(n(3),n(3),'symmetric');
% pbLMI = pbLMI + (bn3>=0);
Qw = blkdiag(dn1+bn2,dn2); %
pbLMI = pbLMI + (blkdiag(dn1,dn2,bn1)>=b*eye(n(2)+n(1)+nG));%
rn1 = sdpvar(n(1),n(1),'symmetric');
pbLMI = pbLMI + (rn1>=0);
rn2 = sdpvar(n(2),n(2),'symmetric');
pbLMI = pbLMI + (rn2>=0);
% rn3 = sdpvar(n(3),n(3),'symmetric');
% pbLMI = pbLMI + (rn3>=0);
Qe = blkdiag(rn1-bn2,rn2); %
pbLMI = pbLMI + (blkdiag(rn1,rn2)<=1*eye(n(1)+n(2)));  
% 
Rphis = [eye(nG) zeros(nG,nphi) zeros(nG,nphi);
        Nvx Nvw Nvw;
        zeros(nphi,nG) eye(nphi) zeros(nphi,nphi);
        zeros(nphi,nG) zeros(nphi,nphi) eye(nphi)];      
    
Mphi = [zeros(nG) Z' -Z' zeros(nG,nphi);
           Z -2*T 2*T+T*Nvw T*Nvw;
          -Z 2*T+Nvw'*T -T*(eye(nphi)+Nvw)-(eye(nphi)+Nvw)'*T -T*Nvw;
          zeros(nphi,nG) Nvw'*T -Nvw'*T zeros(nphi)];  
      
Qphis = Rphis'*Mphi*Rphis;  

lmi11 = (AG+BG*Nux)'*P*(AG+BG*Nux)-P + bn1; 
lmi12 = (AG+BG*Nux)'*P*BG*Nuw;
lmi13 = (AG+BG*Nux)'*P*BG*Nuw;
lmi22 = Nuw'*BG'*P*BG*Nuw + Qw;
lmi23 = Nuw'*BG'*P*BG*Nuw;
lmi33 = Nuw'*BG'*P*BG*Nuw - Qe;

QV = [lmi11  lmi12  lmi13;
      lmi12' lmi22  lmi23;
      lmi13' lmi23' lmi33];

MSTAB = QV + Qphis;   

%D = blkdiag(eye(nG),eye(nphi),eye(nphi));

pbLMI = pbLMI + (MSTAB <= 0) ;%+ (D'*MSTAB*D >= -a*eye(size(MSTAB,1)));%
     
for i = 1:nphi
    lmi111 = 2*T(i,i)*(1/mu(i))*eye(nG);
    lmi112 = Z(i,:)';
    lmi113 = (1/mu(i))*eye(nG);
    lmi122 = deltav1(i)^2;
    lmi123 = zeros(1,nG);
    lmi133 = P;
    
    MSETOR = [lmi111 lmi112 lmi113;
    lmi112' lmi122 lmi123;
    lmi113' lmi123' lmi133];
    pbLMI = pbLMI + (MSETOR>=0);
end

lmi211 = x1bound^2;
lmi212 = [1,0];
lmi222 = P;
MSETOR2 = [lmi211 lmi212;
    lmi212' lmi222];
pbLMI = pbLMI + (MSETOR2>=0);

% critere d'optimisation
critOPTIM = -b;%-b;%trace(Qw(1:n1,1:n1));%(n1+1:n1+n2,n1+1:n1+n2)
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

feasible = min(checkset(pbLMI));
if (feasible >= 0) % && solPb.problem == 0)
    sol.P = double(P);
    sol.Qw =  double(blkdiag(dn1,dn2,bn1,bn2));%double(Qw);%
    sol.Qe = double(blkdiag(rn1,rn2));%double(Qe);%
    sol.T = double(T);
    sol.G = sol.T\double(Z);
else
    sol.P = [];
    sol.Qw = [];
    sol.Qe = [];
    sol.T = [];
    sol.G = [];
end

end