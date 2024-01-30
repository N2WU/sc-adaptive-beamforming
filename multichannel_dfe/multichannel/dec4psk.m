function d=dec4psk(x);

xr=real(x);xi=imag(x);
dr=sign(xr); di=sign(xi); 
d=dr+j*di;
d=d/sqrt(2);
