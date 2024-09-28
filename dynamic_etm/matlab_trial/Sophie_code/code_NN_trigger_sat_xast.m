function [sol,solPb] = code_NN_trigger_sat_xast(sysP,sysC,x1b,vb,mu,xast)

pbLMI = [];
AG = sysP.AG;
BG = sysP.BG2;

nG = sysP.nG;
nlayer = numel(sysC.W)-1;

N = [];
n = zeros(1,nlayer);
W = cell(1,nlayer);
b = cell(1,nlayer);
Nb = [];
for i=1:nlayer+1
    W{i} = sysC.W{i};
    b{i} = sysC.b{i};
    n(i) = size(W{i},1);
    N = blkdiag(N,W{i});
    Nb = [Nb;b{i}];
end
nphi = sum(n(1:nlayer));
Nux = N(nphi+1:end,1:nG);
Nuw = N(nphi+1:end,nG+1:end);
Nub = Nb(nphi+1:end,1);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);
Nvb = N(1:nphi,1);

%%
% Definition des variableset du systeme de LMIs a contruire
% a = sdpvar(1,1);   
% pbLMI = pbLMI + (a>=0);
P = sdpvar(nG,nG,'symmetric');   
pbLMI = pbLMI + (P>=1e-10*eye(size(P,1)));
T = sdpvar(nphi,nphi,'diagonal');
pbLMI = pbLMI + (T>=0);%diag(mu)
Z = sdpvar(nphi,nG,'full');
      
Rphis = [eye(nG) zeros(nG,nphi) zeros(nG,1);
        Nvx Nvw Nvb;
        zeros(nphi,nG) eye(nphi) zeros(nphi,1)];   
    
% Rphis = [eye(nG) zeros(nG,nphi);
%         Nvx Nvw;
%         zeros(nphi,nG) eye(nphi)];     

Mphi = [zeros(nG) -Z' Z';
          -Z zeros(nphi) T;
           Z T -2*T];  
        
Qphis = Rphis'*Mphi*Rphis; 

Rs = [eye(nG) zeros(nG,nphi) zeros(nG,1);
      Nux Nuw Nub;
      zeros(1,nG) zeros(1,nphi) 1];
%   
% Rs = [eye(nG) zeros(nG,nphi);
%       Nux Nuw];  

lmi11 = AG'*P*AG-P;
lmi12 = AG'*P*BG;
lmi13 = -AG'*P*xast+P*xast;
lmi22 = BG'*P*BG;
lmi23 = -BG'*P*xast;
lmi33 = 0;

Qs = [lmi11  lmi12  lmi13;
      lmi12' lmi22  lmi23;
      lmi13' lmi23' lmi33];
  
% Qs = [lmi11  lmi12;
%       lmi12' lmi22];  

Qlyap = Rs'*Qs*Rs;
  
MSTAB =  Qlyap  + Qphis; 

%D = blkdiag(eye(nG), eye(nphi));
pbLMI = pbLMI + (MSTAB <= 0);% + (D'*MSTAB*D>=-a*eye(nG+nphi));

lmi111 = P; 
lmi112 = -P*xast;
lmi122 = xast'*P*xast;
lmi123 = 0;
for i = 1:nphi
    lmi113 = Z(i,:)';
    lmi133 = 2*mu(i)*T(i,i)-mu(i)^(2)*vb(i)^(-2);%mu(i)*T(i,i)*vb(i)^(2);%
    MSETOR = [lmi111 lmi112 lmi113;
    lmi112' lmi122 lmi123;
    lmi113' lmi123' lmi133];
%     MSETOR = [lmi111 lmi113;
%               lmi113' lmi133];

    pbLMI = pbLMI + (MSETOR>= 0);%
end
% 
lmi211 = P;
lmi212 = -P*xast;
lmi213 = [1;0];
lmi222 = xast'*P*xast;
lmi223 = 0;
lmi233 = x1b^2;
MSETOR2 = [lmi211 lmi212 lmi213;
    lmi212' lmi222 lmi223;
    lmi213' lmi223' lmi233];
% MSETOR2 = [lmi211 lmi213;
%     lmi213' lmi233];
%pbLMI = pbLMI + (MSETOR2>= 0);

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