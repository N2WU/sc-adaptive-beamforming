function [del,x]=fdel(v,u);

N=ceil(log10(length(v))/log10(2)); % first 2^N that will do
x=ifft(fft(v,2^N).*conj(fft(u,2^N)));
[m,del]=max(x);


