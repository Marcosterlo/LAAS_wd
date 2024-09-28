clear all
clc
%% parameters
g = 10; % gravitational coefficient
m = 0.15; % mass
l = 0.5; % length
mu = 0.05; % frictional coefficient
dt = 0.02; % sampling period

%% x^+ = AG*x + BG*q
sysP.AG = [1, dt;...
    g/l*dt, 1-mu/(m*l^2)*dt];
% describes how u enters the system
sysP.BG = [0;...
    dt/(m*l^2)];

%% load weights and biases of the NN controller
%fname = '../vehicle_training/Wb_s32_tanh/';
% fname = 'Wb_s32_relu/';
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
%% 
nG = size(sysP.AG,1);
nlayer = numel(W)-1;
N = [];
%b = cell(1,nlayer+1);
n = zeros(1,nlayer+1);
for i=1:nlayer+1
    n(i) = size(W{i},1);
    sysC.W{i} = W{i};
    sysC.b{i} = b{i};
    N = blkdiag(N,W{i});
end
nphi = sum(n(1:nlayer));
Nux = [-0.1 0];
Nuw = N(nphi+1:end,nG+1:end);
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);

Nub = b{3};
Nvb = [b{1};b{2}];
xast = (eye(nG)-sysP.AG-sysP.BG*(Nux+Nuw*(eye(nphi)+Nvw)*Nvx))\(sysP.BG*(Nuw*(eye(nphi)+Nvw)*Nvb+Nub));
uast = (Nux+Nuw*(eye(nphi)+Nvw)*Nvx)*xast + Nuw*(eye(nphi)+Nvw)*Nvb + Nub;
vast = (eye(nphi)+Nvw)*Nvx*xast + (eye(nphi)+Nvw)*Nvb;
wast = vast;

sysP.xast=xast;
sysP.wast=wast;

vb = [1*ones(32,1);1*ones(32,1)];
x1bound = 2.5 -xast(1,1);
for i=1:nphi
    vbound = min(abs(vb-vast),abs(-vb-vast));
end
%% Convex Optimization - compute trigger parameters and ROA

mu = [9e-4*ones(32,1);9e-4*ones(32,1)];

[sol,solPb] = code_NN_trigger_sat_Ch(sysP,sysC,x1bound,vbound,mu); %0.0009
sol
%[sol,solPb] = code_NN_trigger_sat(sysP,sysC,x1bound,vbound,mu);%0.002

%% Simulation results
% Convergent and divergent trajectories
figure(6)
Nstep = 500;
for i=-18:1.8:18
    for j=-36:3.6:36
        x0 = [i,j];
        [x,u,update,sumup] = NN_trigger_w_closedloop_Ch(Nstep,sysP,x0,W,b,vb,sol);
%         [x,u] = NN_trigger_w_closedloop(Nstep,sysP,x0,vb,W,b);
        if abs(x(1,end))>2*abs(xast(1))
            plot(x(1,2:end)-xast(1)*ones(1,Nstep),x(2,2:end),'g-.','LineWidth',1)
            hold on
            plot(x(1,2)-xast(1),x(2,2),'g*')%,'MarkerFaceColor','c')
        else
            plot(x(1,2:end)-xast(1)*ones(1,Nstep),x(2,2:end),'c-','LineWidth',1)
            hold on
            plot(x(1,2)-xast(1),x(2,2),'cs')
        end
    end
end
xlim([-18 18])
ylim([-36 36])
% line([-(2.5-xast(1)) -(2.5-xast(1))],[-36 36])
% line([2.5-xast(1) 2.5-xast(1)],[-36 36])
set(gca,'Ticklabelinterpreter','latex')

%% Update
Nstep = 350;
sumup_traj = [];
sumup_total = zeros(2,1);
figure(1)
for i=1:20 %size(ptsxf,2)
        [x,u,update,sumup] = NN_trigger_w_closedloop_Ch(Nstep,sysP,ptsxf(:,10),W,b,vb,sol);
%        [x,u] = NN_trigger_w_closedloop(Nstep,sysP,x0,vb,W,b);    
         sumup_traj = [sumup_traj sumup];
         
%          k = 1:1:Nstep-2;
%          subplot(2,1,1)
%          stairs(k,x(1,2:end-2),'LineStyle','-','LineWidth',1.2,'Color','b');
%          hold on
%          stairs(k,x(2,2:end-2),'LineStyle','-','LineWidth',1.2,'Color','g');
%          
%          subplot(2,1,2)
%          stairs(k,u(1,1:end-2),'LineStyle','-','LineWidth',1.2,'Color','m');
%          hold on
            subplot(3,1,1)
            aux = ones(2,1);
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
end
sumup_total(1) = mean(sumup_traj(1,2:end));
sumup_total(2) = mean(sumup_traj(2,2:end));
sumup_total/3.5
sum(sumup_total)/7

% 23.0534/22.6147
% 21.1977/20.6724
% 18.6147/17.9740
%% Figures
figure(7)

k = 1:1:Nstep-2;
subplot(3,1,1)
stairs(k,x(1,2:end-2),'LineStyle','-','LineWidth',1.2,'Color','b');
hold on
stairs(k,x(2,2:end-2),'LineStyle','-','LineWidth',1.2,'Color','g');
xlabel('$k$' ,'interpreter','latex')
ylabel('$x_k$','interpreter','latex');
h=legend( '$x_1$','$x_2$','Location', 'northeast');
set(h,'Interpreter','latex');
set(gca,'Ticklabelinterpreter','latex')
grid

ax2 = axes('position',[.757 .816 .163 .145],'Box','on');
axes(ax2)%hold on
stairs(k(300:350),x(1,302:352),'LineStyle','-','LineWidth',1.2,'Color','b');%,'Marker','o', 'MarkerEdgeColor','auto', 'MarkerFaceColor','auto')
hold on
stairs(k(300:350),x(2,302:352),'LineStyle','-','LineWidth',1.2,'Color','g');%,'Marker','o', 'MarkerEdgeColor','auto', 'MarkerFaceColor','auto')
%ylim([-1.4 -0.7])
grid

% control 
subplot(3,1,2)
stairs(k,u(1,1:end-2),'LineStyle','-','LineWidth',1.2,'Color','r');
hold on
xlabel('$k$' ,'interpreter','latex')
ylabel('$u_k$','interpreter','latex');
% h=legend( '$u_k$','Location', 'northeast');
% set(h,'Interpreter','latex');
set(gca,'Ticklabelinterpreter','latex')
grid

ax3 = axes('position',[.683 .272 .163 .145],'Box','on');
axes(ax3)
hold on
stairs(k(300:350),u(1,300:350),'LineStyle','-.','LineWidth',1.2,'Color','r');%,'Marker','o', 'MarkerEdgeColor','auto', 'MarkerFaceColor','auto')
%ylim([-1.4 -0.7])
grid

% inter-events
subplot(3,1,3)
aux = ones(2,1);
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
xlabel('$k$' ,'interpreter','latex')
ylabel('Inter-events','interpreter','latex');
h=legend( '$\hat{\omega}^1$','$\hat{\omega}^2$','Location', 'northeast','Orientation','horizontal');
set(h,'Interpreter','latex');
set(gca,'Ticklabelinterpreter','latex')
grid

xlabel('$x_1-x_{\ast,1}$' ,'interpreter','latex')
ylabel('$x_2-x_{\ast,2}$','interpreter','latex');
h=legend([p1,p2],{'$\mathcal{E}(P,1)$','$\mathcal{E}(X,1)$'},'Location', 'northeast','Orientation','horizontal');
set(h,'Interpreter','latex');
set(gca,'Ticklabelinterpreter','latex')
grid
%% 
figure(2)
P0 = 2*[0.1458    0.0027;
    0.0027    0.0045];
[U, D, V] = svd(sol.P(1:2,1:2))

z = 1/sqrt(D(1,1));
y = 1/sqrt(D(2,2));
theta = [0:1/20:2*pi+1/20];

state(1,:) = z*cos(theta);
state(2,:) = y*sin(theta);

X = V * state;

lx1 = max(X(1,:));
lx2 = max(X(2,:));
z

p2 = plot(X(1,:),X(2,:),'Color','g','LineWidth',1.2,'LineStyle' ,'-');
hold on

x1 = linspace(-lx1,lx1,150);
x2 = linspace(-lx2,lx2,150);
pts = [];
ptsx = [];
for i = 1:150
    for j = 1:150
        if ([x1(i),x2(j)]*sol.P(1:2,1:2)*[x1(i);x2(j)] <=1) 
           plot(x1(i),x2(j),'.','color','g');
           hold on
%            plot(x1(i)+xast(1,1),x2(j)+xast(2,1),'.','color','b');
%            hold on
           pts = [pts, [x1(i);x2(j)]];
           ptsx = [ptsx, [x1(i);x2(j)]+xast];
        end
    end
end

ptsf = [];
ptsxf = [];
for i=1:170:size(pts,2)
    ptsf = [ptsf,pts(:,i)];
    ptsxf = [ptsxf,ptsx(:,i)];
end
% (2.5598 13.8308) (2.5437 15.7891) (2.5319 17.0292) 
%% Set S
x1 = linspace(-36,36,1e3)';
for i=1:n(1)
    plot(x1,sol.G(i,2)\(vb(i)-vast(i)-sol.G(i,1)*x1),'color',[1 .4 0]) 
    hold on;
    plot(x1,sol.G(i,2)\(-vb(i)-vast(i)-sol.G(i,1)*x1),'color',[1 .4 0])
end
for i=n(1)+1:n(1)+n(2)
    plot(x1,sol.G(i,2)\(vb(i)-vast(i)-sol.G(i,1)*x1),'color', [1 .55 0]) 
    hold on;
    plot(x1,sol.G(i,2)\(-vb(i)-vast(i)-sol.G(i,1)*x1),'color', [1 .55 0])
end
%% Linear region

subplot(1,2,2)
x1 = linspace(-7,7,1e3)';
for i=1:n(1)
    plot(x1,sol.G(i,2)\(vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2)),'color',[1 .4 0]) 
    hold on;
    plot(x1,sol.G(i,2)\(-vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2)),'color',[1 .4 0])
end
for i=n(1)+1:n(1)+n(2)
    plot(x1,sol.G(i,2)\(vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2)),'color', [1 .55 0]) 
    hold on;
    plot(x1,sol.G(i,2)\(-vb(i)-vast(i)-sol.G(i,1)*x1+sol.G(i,1)*xast(1)+sol.G(i,2)*xast(2)),'color', [1 .55 0])
end
%% Constraints
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
        plot(k,s2(i),'bs')
        hold on
    end
end