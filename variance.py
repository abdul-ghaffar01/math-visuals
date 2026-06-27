import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

plt.scatter(x, y, s=80)

# Mean point
plt.scatter(np.mean(x), np.mean(y),
            color='red',
            s=120,
            label='Mean')

mx = np.mean(x)
my = np.mean(y)

for i in range(len(x)):
    plt.plot([mx, x[i]],
             [my, y[i]],
             '--',
             color='gray')

plt.grid(True)
plt.axis('equal')
plt.legend()

plt.show()