import numpy as np
import matplotlib.pyplot as plt

# Creating the first array
v = np.array([3, 2])

print(v)

# Creating the plane
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# plt.gca().set_aspect(1)
plt.axis('equal')
plt.grid()

plt.axhline(y=0)
plt.axvline(x=0)


# Drawing the numpy array
plt.arrow(0, 0, v[0], v[1], length_includes_head=True)


matrix = np.array([
    [2, 0],
    [0, 2]
])

nm = matrix @ v
print(nm)
plt.arrow(0, 0, nm[0], nm[1], length_includes_head=True)


plt.show()