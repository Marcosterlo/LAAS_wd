function [theta] = avalia_theta2( nlayer, theta, x, v, wh, T, G)

erro = zeros(1,nlayer);

for i = 1:nlayer
    
    erro(i) = (v{i}-wh{i})'*T{i}*(v{i}-wh{i}-G{i}*x);
end

pos = [];
erro_pos = [];
erro_neg = [];
for i=1:nlayer
    if theta(i)> 0 
        pos = [pos, theta(i)];
        erro_pos = [erro_pos, erro(i)];
    else
        erro_neg = [erro_neg, erro(i)];
    end 
end
for i=1:size(pos,2)
    if erro_pos(i)> theta(i)
       if sum(erro_neg)<sum(erro_pos)
            theta = zeros(1,nlayer);
            break;
       end
    end
end
end
