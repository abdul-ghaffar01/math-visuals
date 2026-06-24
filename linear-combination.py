import matplotlib.pyplot as plt
import numpy as np


# Let there are 2 vectors, u and v now if we scale u with a and v with b what kind of combinations we can get,
# One step is linear combination and all the possile steps are linear span

# Creating vector u
u = np.array([3, 2])

# creating vector v
v = np.array([2, 3])

# Let a and b are two constants
a = 4
b = 5

# So the linear combination will be
lc = (a * u) + (b * v)
print(lc)
# lets plot this linear combination
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# Created two lines to make it look like a 2D plane
plt.axhline(y=0)
plt.axvline(x=0)

# ploting the linear combination
plt.arrow(0, 0, lc[0], lc[1])

plt.grid()
plt.show()

