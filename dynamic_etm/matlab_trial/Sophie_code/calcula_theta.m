function [theta] = calcula_theta(n, nlayer,deltav1, x, wh, W, b, T, G)

nG = size(x,1);
w = cell(nlayer,1);
v = cell(nlayer,1);
erro_next = cell(1,nlayer);
for i = 1:nlayer
    w{i} = zeros(n(i),1);
    v{i} = zeros(n(i),1);
    erro_next{i} = zeros(1,i);
end

wopt = cell(1,nlayer);
v{1} = W{1}*x + b{1};
w{1} = sign(v{1}).*min(abs(v{1}),deltav1);
erro_next{1} = (v{1}-wh{1})'*T{1}*(v{1}-wh{1}-G{1}*x);
        
wopt{1} = [w{1},wh{1}];
for i = 2:nlayer
    for j = 1:2
        v{i} = W{i}*wopt{i-1}(:,j) + b{i};
        w{i} = sign(v{i}).*min(abs(v{i}),deltav1*ones(n(i),1));
        erro_next{i}(:,j) = (v{i}-wh{i})'*T{i}*(v{i}-wh{i}-G{i}*x);
        
        wopt{i} = [wopt{i},w{i}];
    end
end

delta  = [];
for i= 2:nlayer
    aux = [];
    for j = 1:size(erro_next{i},2)
        aux = [aux, erro_next{1}-erro_next{i}(:,j)];
    end
    delta = [delta,mean(aux)];
end

theta = inv([1 1;1 -1])*[0;delta(1)];
%theta = inv([1 1 1;1 -1 0;1 0 -1])*[0;delta(1);delta(2)];
end