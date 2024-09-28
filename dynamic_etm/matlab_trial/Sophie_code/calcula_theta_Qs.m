function [theta] = calcula_theta_Qs(n, nlayer, x, wh, W, b, vlim, Qe, Qw, Qwx)

nG = size(x,1);
v = cell(nlayer,1);
w = cell(nlayer,1);
erro_next = cell(1,nlayer);
for i = 1:nlayer
    v{i} = zeros(n(i),1);
    w{i} = zeros(n(i),1);
    erro_next{i} = zeros(1,i);
end

wopt = cell(1,nlayer);
v{1} = W{1}*x + b{1};
w{1} = sign(v{1}).*min(abs(v{1}),vlim{1});
            
erro_next{1} = (wh{1}-w{1})'*Qe{1}*(wh{1}-w{1}) - w{1}'*Qw{1}*w{1} - x'*Qwx{1}*x;
wopt{1} = [w{1},wh{1}];
for i = 2:nlayer
    for j = 1:2
        v{i} = W{i}*wopt{i-1}(:,j) + b{i};
        w{i} = sign(v{i}).*min(abs(v{i}),vlim{i});
        erro_next{i}(:,j) = (wh{i}-w{i})'*Qe{i}*(wh{i}-w{i}) - w{i}'*Qw{i}*w{i} - wopt{i-1}(:,j)'*Qwx{i}*wopt{i-1}(:,j);
        wopt{i} = [wopt{i},w{i}];
    end
end

delta  = [];
for i = 1:nlayer
    for j= i+1:nlayer
        aux = [];
        for s = 1:size(erro_next{j},2)
            aux = [aux, erro_next{i}-erro_next{j}(:,s)];
        end
        delta = [delta, min(aux)];%min(aux)*min(abs(aux))/max(abs(aux)) + max(aux)*min(abs(aux))/max(abs(aux))
    end
end
theta = inv([1 1;1 -1])*[0;delta(1)];
%    theta(:,k) = inv([1 1 1;1 -1 0;1 0 -1])*[0;delta(1);delta(2)];
end