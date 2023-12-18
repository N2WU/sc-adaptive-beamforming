import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Plot a sin curve using the x and y axes.
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
z = np.arange(10)
my_array = np.zeros((len(y),len(z)))

for i in range(len(z)):
    my_array[:,i] = y

print(np.shape(my_array))

for i in range(len(z)):
    ax.plot(np.arange(len(my_array[:,i])), my_array[:,i], zs=i, zdir='y')
ax.view_init(elev=20., azim=-35, roll=0)
plt.ylabel('time (sec)')
plt.xlabel('delay (ms)')
plt.show()