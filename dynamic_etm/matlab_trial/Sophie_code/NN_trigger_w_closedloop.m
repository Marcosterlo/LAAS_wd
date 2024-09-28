function [x,u] = NN_trigger_w_closedloop(Nstep,sysP,x0,deltav1,W,b)

AG = sysP.AG;
BG = sysP.BG;

nlayer = numel(W)-1;
n = zeros(1,nlayer);
for i=1:nlayer
    n(i) = size(W{i},1);
end
vb{1} = deltav1(1:32);
vb{2} = deltav1(33:64);

Nx = numel(x0);

x = zeros(Nx,Nstep);
x(:,2) = x0;

w = cell(nlayer,1);
v = cell(nlayer,1);
for i = 1:nlayer
    w{i,1} = zeros(n(i),1);
    v{i,1} = zeros(n(i),1);
end

Nu = size(W{end},1);
u = zeros(Nu,Nstep);

% Simulate System
for k = 2:Nstep
       
    v{1,k} = W{1}*x(:,k) + b{1};
    w{1,k} = sign(v{1,k}).*min(abs(v{1,k}),vb{1});%tanh(v{2,k});%
    
    for i = 2:nlayer
        
        v{i,k} = W{i}*w{i-1,k} + b{i};
        w{i,k} = sign(v{i,k}).*min(abs(v{i,k}),vb{i});%tanh(v{i+1,k});%
        
    end
    
    u(:,k) = W{end}*w{end,k}+b{end};
    
    %     x(:,k+1) = x(:,k)+[x(2,k);...
    %         g/l*sin(x(1,k))-mu/(m*l^2)*x(2,k)+1/(m*l^2)*u(:,k)]*dt;
    %     x(:,k+1) = sysP.A*x(:,k)+sysP.B*u(:,k)+sysP.B2*(x(1,k)-sin(x(1,k)));
    x(:,k+1) = AG*x(:,k)+BG*u(:,k);%+Bq*(x(1,k)-sin(x(1,k)));
    
end
end