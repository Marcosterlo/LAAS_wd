function [theta] = avalia_theta_Qs(nlayer, theta, x, w, wh, Qe, Qw, Qwx)

erro = zeros(1,nlayer);

erro(1) = (wh{1}-w{1})'*Qe{1}*(wh{1}-w{1}) - w{1}'*Qw{1}*w{1} - x'*Qwx{1}*x;

for i = 2:nlayer
    
    erro(i) = (wh{i}-w{i})'*Qe{i}*(wh{i}-w{i}) - w{i}'*Qw{i}*w{i} - wh{i-1}'*Qwx{i}*wh{i-1};

end

for i=1:nlayer
    if sum (erro)>0
       theta = zeros(1,nlayer);
       break;
    end
end
end
