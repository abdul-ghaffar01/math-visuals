import matplotlib.pyplot as plt
import numpy as np

# Task:
# Create an object on the plane and then rotate it using matrix transformation

# setting the limit
plt.xlim(-20, 20)
plt.ylim(-20, 20)

# Drawing x and y axises
plt.axhline(y=0)
plt.axvline(x=0)

plt.gca().set_aspect('equal')

line = np.array([
    [1, 2],
    [10, 3]
])

x = [p[0] for p in line]
y = [p[1] for p in line]

# Drawing a line in the first quadrant
plt.plot(x, y, color='red')

# Applying 90 deg rotation transformation on the line
rotation = np.array([
    [0, -1],
    [1, 0]
])

lineRotated = np.matmul(line, rotation.T)
new_x = [p[0] for p in lineRotated]
new_y = [p[1] for p in lineRotated]
plt.plot(new_x, new_y, color='blue')

plt.grid()
plt.show()