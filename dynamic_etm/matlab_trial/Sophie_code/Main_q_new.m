clear all
clc
%% parameters
g = 10; % gravitational coefficient
m = 0.15; % mass
l = 0.5; % length
mu = 0.05; % frictional coefficient
dt = 0.02; % sampling period

%% x^+ = AG*x + BG1*q + BG2*u
sysP.AG = [1,      dt;...
    g/l*dt, 1-mu/(m*l^2)*dt];
% describes how q enters the system
sysP.BG1 = [0;...
    -g/l*dt];
% describes how u enters the system
sysP.BG2 = [0;...
    dt/(m*l^2)];
sysP.nG = size(sysP.AG, 1);
nu = 1;
nq = 1;

%  v_Delta = CG*xG + [DG1 DG2]*[q; u] = xG
CG = [1, 0];
DG1 = 0;
DG2 = 0;

%% load weights and biases of the NN controller
%fname = '../vehicle_training/Wb_s32_tanh/';
% fname = 'Wb_s32_tanh/';
% load([fname 'W1.csv'])
% load([fname 'W2.csv'])
% load([fname 'W3.csv'])
load('sat_W3_ini_8.mat')
W{1} = W1;
W{2} = W2;
W{3} = W3;
b{1} = b1;
b{2} = b2;
b{3} = b3;

%load('file_W3_ini.mat')

% for ii=0.2:0.1:0.4

nlayer = numel(W)-1;

%b = cell(1,nlayer+1);
n = zeros(1,nlayer+1);
for i=1:nlayer+1
    n(i) = size(W{i},1);
    sysC.W{i} = W{i};
%    b{i} = zeros(n(i),1);%
    sysC.b{i} = b{i};
end
nphi = sum(n(1:nlayer));

%% Define IQCs for the nonlinearity x1 - sin(x1)
% x1 - sin(x1) is slope-restricted in [0, 2] globally ---> off-by-one IQC

% x1 - sin(x1) is sector-bounded in [0, 1.22] globally ---> sector IQC
% x1 - sin(x1) is sector-bounded in [0, 1] locally on x1 in [-pi,pi]
% x1 - sin(x1) is sector-bounded in [0, 0.7606] locally on x1 in [-2.5,2.5]
% define the filter for off-by-one IQC
%L_slope = 2;
% x1bound = 0.73;
% L_slope = 1 - cos(x1bound);
% m_slope = 0;
% Apsi = 0;
% sysP.nxi = size(Apsi,1);
% Apsi = Apsi*dt + eye(sysP.nxi);
% Bpsi1 = -L_slope;
% Bpsi2 = 1;
% Bpsi1 = Bpsi1*dt;
% Bpsi2 = Bpsi2*dt;
% Cpsi = [1; 0];
% Dpsi1 = [L_slope; -m_slope];
% Dpsi2 = [-1; 1];

%% construct the extended system
% sysP.A = [sysP.AG, zeros(sysP.nG,sysP.nxi);...
%     Bpsi1*CG, Apsi];
% sysP.Bq = [sysP.BG1;
%     Bpsi1*DG1+Bpsi2];
% sysP.Bu = [sysP.BG2;
%     Bpsi1*DG2];
% sysP.C = [Dpsi1*CG, Cpsi];
% sysP.Dq = Dpsi1*DG1+Dpsi2;
% sysP.Du = Dpsi1*DG2;
% sysP.nzeta = sysP.nG + sysP.nxi;
%------------------------------------------------------------------------
nG = 2;
nlayer = numel(W)-1;

N = [];
n = zeros(1,nlayer);
for i=1:nlayer+1
    n(i) = size(W{i},1);
    N = blkdiag(N,W{i});
end
nphi = sum(n(1:nlayer));
Nux = N(nphi+1:end,1:nG);
Nuw = N(nphi+1:end,nG+1:end);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);

Nub = b{3};
Nvb = [b{1};b{2}];
xast = (eye(nG)-sysP.AG-sysP.BG2*(Nux+Nuw*(eye(nphi)+Nvw)*Nvx))\(sysP.BG2*(Nuw*(eye(nphi)+Nvw)*Nvb+Nub));
%xast = (eye(nG)-sysP.AG-sysP.BG2*(Nux+Nuw*(eye(nphi)+Nvw)*Nvx)-sysP.BG1*[1 0])\(sysP.BG2*(Nuw*(eye(nphi)+Nvw)*Nvb+Nub)-sysP.BG2*[1 0]*[-1;0]);
uast = (Nux+Nuw*(eye(nphi)+Nvw)*Nvx)*xast + Nuw*(eye(nphi)+Nvw)*Nvb + Nub;
vast = (eye(nphi)+Nvw)*Nvx*xast + (eye(nphi)+Nvw)*Nvb;
wast = vast;
sol.xast=xast;
sol.wast=wast;
%% Convex Optimization - compute trigger parameters and ROA
%0.3
vb = [1*ones(32,1);1*ones(32,1)];
x1bound = 2.5;% -xast(1,1);
for i=1:nphi
    vbound = min(abs(vb-vast),abs(-vb-vast));
end
for i=0.010:0.001:0.015
 %   for j=5:0.25:8
    mu = [9e-4*ones(32,1);9e-4*ones(32,1)];%2e-3 3e-3
%    [sol,solPb] = code_NN_trigger_sat(sysP,sysC,x1bound,vbound,mu);

%     [sol,solPb] = code_NN_trigger_sat_xast(sysP,sysC,x1bound,vb,mu,xast);   
    [sol,solPb] = code_NN_trigger_sat_Qs(sysP,sysC,x1bound,vbound,mu); %0.0009
%    [sol,solPb] = code_NN_trigger_sat_Qs_2(sysP,sysC,x1bound,deltav1,mu);%100
%     mu = [9e-3*ones(32,1);9e-2*ones(32,1)];%0.0009
%     [sol,solPb] = code_NN_trigger_sat_q(sysP,sysC,x1bound,deltav1,mu);
%    [sol,solPb] = code_NN_trigger_sat(sysP,sysC,x1bound,deltav1,mu);
    sol.P
 %   end
end
sol
% end

%% plot results
% Simulation results
Nstep = 500;
[vet val] = eig(sol.P(1:2,1:2));
x0 = 1/sqrt(val(1,1))*vet(:,1);
% x0 = 1/sqrt(val(2,2))*vet(:,2);
x0 = [0.3758 -1.485];

Nstep = 350;
sumup_traj = [];
sumup_total = zeros(2,1);
for i=1:101
% for i=-7:0.35:7
%     for j=-7:0.35:7
%         x0 = [i,j];
        [x,u,update,sumup] = NN_trigger_w_closedloop_Qs(Nstep,sysP,x0+xast,W,b,vb,sol);
%        [x,u] = NN_trigger_w_closedloop(Nstep,sysP,x0,vb,W,b);    
         sumup_traj = [sumup_traj sumup];
%         if abs(x(1,end))>1
%             plot(x(1,2:end)-xast(1)*ones(1,Nstep),x(2,2:end),'g-.','LineWidth',1)
%             hold on
%             plot(x(1,2)-xast(1),x(2,2),'g*')%,'MarkerFaceColor','c')
%         else
%             plot(x(1,2:end)-xast(1)*ones(1,Nstep),x(2,2:end),'k-','LineWidth',1)
%             hold on
%             plot(x(1,2)-xast(1),x(2,2),'ks')
% %             hold on 
% %             plot(x(1,end)-xast(1),x(2,end),'bo','LineWidth',1)
%         end
%     end
end
xlim([-7 7])
ylim([-7 7])
% axis square
set(gca,'Ticklabelinterpreter','latex')
line([-(2.5-xast(1)) -(2.5-xast(1))],[-7 7])
line([2.5-xast(1) 2.5-xast(1)],[-7 7])
sumup_total(1) = mean(sumup_traj(1,2:end));
sumup_total(2) = mean(sumup_traj(2,2:end));
sumup_total/5
sum(sumup_total)/10

% states
figure(1)

k = 1:1:Nstep;
subplot(3,1,1)
stairs(k,x(1,2:end),'LineStyle','-','LineWidth',1.2,'Color','b');
hold on
stairs(k,x(2,2:end),'LineStyle','-','LineWidth',1.2,'Color','g');
xlabel('$[s]k$' ,'interpreter','latex')
ylabel('$x_k$','interpreter','latex');
h=legend( '$x_1$','$x_2$','Location', 'northeast');
set(h,'Interpreter','latex');
grid
% 
% ax2 = axes('position',[.757 .816 .163 .145],'Box','on');
% axes(ax2)
% %hold on
% %stairs(k(50:100),x(1,50:100),'LineStyle','-','LineWidth',1.2,'Color','b');%,'Marker','o', 'MarkerEdgeColor','auto', 'MarkerFaceColor','auto')
% hold on
% stairs(k(5:25),x(2,5:25),'LineStyle','-.','LineWidth',1,'Color','g');
% %ylim([-1.4 -0.7])
% grid

% control 
subplot(3,1,2)
stairs(k,u(1,1:end),'LineStyle','-','LineWidth',1.2,'Color','r');
hold on
xlabel('$k[s]$' ,'interpreter','latex')
ylabel('$u_k$','interpreter','latex');
% h=legend( '$u_k$','Location', 'northeast');
% set(h,'Interpreter','latex');
grid

% ax3 = axes('position',[.683 .272 .163 .145],'Box','on');
% axes(ax3)
% hold on
% stairs(k(100:120),u(1,100:120),'LineStyle','-.','LineWidth',1.2,'Color','r');%,'Marker','o', 'MarkerEdgeColor','auto', 'MarkerFaceColor','auto')
% %ylim([-1.4 -0.7])
% grid

% inter-events
% 
% figure(2)
% subplot(2,1,1)
% aux = 1;
% for i = 2:Nstep
%     if update(i)~= 0
%         stem((i-2)*dt, i-aux-1,'b.','MarkerFaceColor','auto');
%         aux = i;
%         hold on
%     end
% end
% xlabel('$k$' ,'interpreter','latex')
% ylabel('Inter-events','interpreter','latex');
% grid

%figure(2)
subplot(3,1,3)
aux = zeros(2,1);
cor = {'c','m'};
fig = {'s','^'};
lin = {'-','--'};
for k = 2:Nstep
    for i = 1:2
        if update(i,k) == 1
%            subplot(2,1,i)
            stem(k-2, k-aux(i)-1,'LineStyle',lin{i},'Color',cor{i},'Marker',fig{i},'MarkerSize',3, 'MarkerFaceColor','auto');
            aux(i) = k;
            hold on
        end
    end
end

% plot(ScopeData1{1}.Values.Time, ScopeData1{1}.Values.Data,'Color','b','LineWidth',1)
% hold on
% plot(ScopeData1{2}.Values.Time, ScopeData1{2}.Values.Data,'Color','r','LineWidth',1)
% for i=1:size(trigger_instants.signals.values,1)
%     if trigger_instants.signals.values(i)==1
%         plot(ScopeData1{1}.Values.Time(i), ScopeData1{1}.Values.Data(i),'Marker','s','MarkerSize',3, 'MarkerEdgeColor','b', 'MarkerFaceColor','w');
%         hold on
%     end
% end

P = cell(2,1);
P{1} = sol.P(1:2,1:2)-sol.P(1:2,3)*inv(sol.P(3,3))*sol.P(3,1:2);
P{2} = sol.P(3,3)-sol.P(3,1:2)*inv(sol.P(1:2,1:2))*sol.P(1:2,3);

figure(2)
x1 = linspace(-1.5,1.5,150);
x2 = linspace(-6,6,150);
pts = [];
ptsx = [];
for i = 1:150
    for j = 1:150
        if (([x1(i);x2(j)]-xast)'*sol.P(1:2,1:2)*([x1(i);x2(j)]-xast) <=1) 
       %     figure(2)
           subplot(1,2,1)
           plot(x1(i)-xast(1,1),x2(j)-xast(2,1),'.','color','b');
           hold on
%            subplot(1,2,2)
%            plot(x1(i),x2(j),'.','color','r');
%            hold on
           pts = [pts, ([x1(i);x2(j)]-xast)];
           ptsx = [ptsx, [x1(i);x2(j)]];
        end
    end
end

ptsf = [];
ptsxf = [];
for i=1:112:size(pts,2)
    ptsf = [ptsf,pts(:,i)];
    ptsxf = [ptsxf,ptsx(:,i)];
end

for i=1:1
    [U D V] = svd(sol.P(1:2,1:2));
    
    z = 1/sqrt(D(1,1));
    y = 1/sqrt(D(2,2));
    theta = [0:1/20:2*pi+1/20];
    
    state(1,:) = z*cos(theta);
    state(2,:) = y*sin(theta);
    
    X = V * state;
    %subplot(1,2,i)
    hold on
    plot(X(1,:),X(2,:),'Color','b','LineWidth',1.2,'LineStyle' ,'-');
    max(X(1,:))
    max(X(2,:))
end

lim = [0.12,0.9,3.0662e+03];
passo = 75;
[c,PTS] = GRID2(sol.P,lim,passo);

%---------------------------------------------------------------------------------- 
subplot(1,2,1)
x1 = linspace(-7,7,1e3)';
for i=1:n(1)
    plot(x1,(vb(i)-vast(i)-sol.G(i,1)*x1)/sol.G(i,2),'color',[1 .4 0]) 
    hold on;
    plot(x1,(-vb(i)-vast(i)-sol.G(i,1)*x1)/sol.G(i,2),'color',[1 .4 0])
end
for i=n(1)+1:n(1)+n(2)
    plot(x1,(vb(i)-vast(i)-sol.G(i,1)*x1)/sol.G(i,2),'color', [1 .55 0]) 
    hold on;
    plot(x1,(-vb(i)-vast(i)-sol.G(i,1)*x1)/sol.G(i,2),'color', [1 .55 0])
end
%%%--------------------------------
subplot(1,2,2)
x1 = linspace(-7,7,1e3)';
for i=1:n(1)
    plot(x1,(vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2))/sol.G(i,2),'color',[1 .4 0]) 
    hold on;
    plot(x1,(-vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2))/sol.G(i,2),'color',[1 .4 0])
end
for i=n(1)+1:n(1)+n(2)
    plot(x1,(vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2))/sol.G(i,2),'color', [1 .55 0]) 
    hold on;
    plot(x1,(-vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2))/sol.G(i,2),'color', [1 .55 0])
end
%--------------------------------------------------------------
for k=1:200
    v1 = W{1}*ptsxf(:,k) + b{1};
    w1 = sign(v1).*min(abs(v1),vb(1:32));
    v2 = W{2}*w1 + b{2};
    w2 = sign(v2).*min(abs(v2),vb(33:64));
    for i=1:n(1)
        j = i+n(1);
        s1(i) = (v1(i)-w1(i))'*sol.T(i,i)*(sol.G(i,1:2)*ptsf(:,k)-(w1(i)-wast(i)));
        s2(i) = (v2(i)-w2(i))'*sol.T(j,j)*(sol.G(j,1:2)*ptsf(:,k)-(w2(i)-wast(j)));
        g1ub(i) = sol.G(i,1:2)*ptsf(:,k)-(vb(i)-vast(i));
        g1lb(i) = (-vb(i)-vast(i))-sol.G(i,1:2)*ptsf(:,k);
        g2ub(i) = sol.G(j,1:2)*ptsf(:,k)-(vb(j)-vast(j));
        g2lb(i) = (-vb(j)-vast(j))-sol.G(j,1:2)*ptsf(:,k);
        figure(4)
        plot(k,g2lb(i),'bs')
        hold on
    end
end
%1.2090
%5.5883


