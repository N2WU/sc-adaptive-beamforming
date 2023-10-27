delta = 0.05;
M = 12;
theta = 45;
a = steering_vec(theta,delta,M)

function a = steering_vec(theta,delta,M)
    a = exp(-1j*2*pi*sind(theta).'*delta*(0:M-1)).';    
end