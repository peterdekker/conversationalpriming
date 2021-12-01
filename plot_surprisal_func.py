# import libraries
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')

# creating the test data
x = np.linspace(0.001, 1.0, 1000)
y=-np.log2(x)

# rendering the chart

fig,ax= plt.subplots()
plt.style.use('ggplot')
ax.plot(x,y);
ax.set_title('Surprisal')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis');
plt.savefig("surprisal.png", dpi=300)