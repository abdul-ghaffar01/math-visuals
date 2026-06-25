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

plt.plot(3, 2, marker='o', color='red')
plt.show()