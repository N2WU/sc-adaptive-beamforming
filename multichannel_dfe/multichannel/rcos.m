% raised cosine;

function p=rcos(alpha,Ns,trunc);

% alpha = roll-off factor
% Ns = samples per symbol;
% trunc =  left/right truncation length in sb. int.

tn=(-trunc*Ns:trunc*Ns)/Ns;
p=sin(pi*tn)./(pi*tn).*cos(alpha*pi*tn)./(1-4*alpha^2*tn.^2);
f0=find(p==Inf|p==-Inf);p(f0)=zeros(size(f0));
p(Ns*trunc+1)=1;

