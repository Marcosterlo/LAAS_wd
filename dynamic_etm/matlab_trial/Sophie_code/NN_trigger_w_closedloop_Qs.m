function [x,u,update,sumup] = NN_trigger_w_closedloop_Qs(Nstep,sysP,x0,W,b,deltav1,sol)
% Simulate system for N steps from the initial condition x0.

AG = sysP.AG;
BG = sysP.BG;
xast = sysP.xast;
wast = sysP.wast;

% Initialize outputs
nlayer = numel(W)-1;
n = zeros(1,nlayer);
for i=1:nlayer
    n(i) = size(W{i},1);
end
nG = size(sysP.AG,1);
Qe = cell(1,nlayer);
Qw = cell(1,nlayer);
for i = 1:nlayer
    Qe{i} = sol.Qe(sum(n(1:i-1))+1:sum(n(1:i)),sum(n(1:i-1))+1:sum(n(1:i)));
    Qw{i} = sol.Qw(sum(n(1:i-1))+1:sum(n(1:i)),sum(n(1:i-1))+1:sum(n(1:i)));
end
Qwx = cell(1,nlayer);
Qwx{1} = sol.Qw(sum(n(1:2))+1:sum(n(1:2))+nG,sum(n(1:2))+1:sum(n(1:2))+nG);
Qwx{2} = sol.Qw(sum(n(1:2))+nG+1:end,sum(n(1:2))+nG+1:end);

Nx = numel(x0);

x = zeros(Nx,Nstep);
x(:,2) = x0;

vb = cell(nlayer,1);
vb{1} = deltav1(1:32);
vb{2} = deltav1(33:64);

v = cell(nlayer,1);
w = cell(nlayer,1);
wh = cell(nlayer,1);
erro = zeros(nlayer,Nstep);
for i = 1:nlayer
    v{i,1} = zeros(n(i),1);
    w{i,1} = zeros(n(i),1);
    wh{i,1} = zeros(n(i),1);
end

Nu = size(W{end},1);
u = zeros(Nu,Nstep);

update = zeros(nlayer,Nstep);
sumup = zeros(nlayer,1);
a=0;
% Simulate System
for k = 2:Nstep
    
    v{1,k} = W{1}*x(:,k) + b{1};
    
    w{1,k} = sign(v{1,k}).*min(abs(v{1,k}),vb{1});%tanh(v{1,k});%
    
    erro(1,k) = (wh{1,k-1}-w{1,k})'*Qe{1}*(wh{1,k-1}-w{1,k}) - (w{1,k}-wast(1:n(1)))'*Qw{1}*(w{1,k}-wast(1:n(1))) - (x(:,k)-xast)'*Qwx{1}*(x(:,k)-xast);
    if (erro(1,k) > 0) || (k==Nstep)
        wh{1,k} = w{1,k};
        update(1,k) = 1;
    else
        wh{1,k} = wh{1,k-1};
        update(1,k) = 0;
    end
    
    for i = 2:nlayer
        
        v{i,k} = W{i}*wh{i-1,k} + b{i};
        
        w{i,k} = sign(v{i,k}).*min(abs(v{i,k}),vb{i});%
        
        erro(i,k) = (wh{i,k-1}-w{i,k})'*Qe{i}*(wh{i,k-1}-w{i,k}) - (w{i,k}-wast(n(1)+1:end))'*Qw{i}*(w{i,k}-wast(n(1)+1:end)) - (wh{i-1,k}-wast(1:n(1)))'*Qwx{i}*(wh{i-1,k}-wast(1:n(1)));
        if (erro(i,k) > 0) ||  (k==Nstep)
            wh{i,k} = w{i,k};
            update(i,k) = 1;
        else
            wh{i,k} = wh{i,k-1};
            update(i,k) = 0;
        end
    end
    
    subplot(3,1,1)
    plot(k-2,v{1,k}(1),'ro','MarkerFaceColor','r')
    hold on
    plot(k-2,w{1,k}(1),'bs','MarkerFaceColor','b')
    hold on
    if v{2,k}(1) ~= w{2,k}(1)
        subplot(3,1,2)
        plot(k-2,v{2,k}(1),'ro','MarkerFaceColor','r')
        hold on
        plot(k-2,w{2,k}(1),'bs','MarkerFaceColor','b')
    else
        subplot(3,1,2)
        plot(k-2,v{2,k}(1),'ro','MarkerFaceColor','r')
        hold on
        plot(k-2,w{2,k}(1),'bs','MarkerFaceColor','b')
    end
    
    u(:,k) = W{end}*wh{end,k}+b{end};
    
    %     x(:,k+1) = x(:,k)+[x(2,k);...
    %         g/l*sin(x(1,k))-mu/(m*l^2)*x(2,k)+1/(m*l^2)*u(:,k)]*dt;
    x(:,k+1) = AG*x(:,k)+BG*u(:,k);%+Bq*(x(1,k)-sin(x(1,k)));
    
end
for i=1:nlayer
    sumup(i) = sum(update(i,:));
end
% subplot(3,1,1)
% xlabel('$k$' ,'interpreter','latex')
% h=legend( '$\omega^1_1$','$\mathtt{sat}(\omega^1_1)$','Location', 'northeast','Orientation','horizontal');
% set(h,'Interpreter','latex');
% set(gca,'Ticklabelinterpreter','latex')
% grid
% xlim([0 25])
% subplot(3,1,2)
% xlabel('$k$' ,'interpreter','latex')
% h=legend( '$\omega^2_1$','$\mathtt{sat}(\omega^2_1)$','Location', 'northeast','Orientation','horizontal');
% set(h,'Interpreter','latex');
% set(gca,'Ticklabelinterpreter','latex')
% grid
% xlim([0 25])
end




