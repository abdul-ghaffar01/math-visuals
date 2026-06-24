import matplotlib.pyplot as plt

# set the limit 
plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.gca().set_aspect('equal')

# Points to start the plane
plt.axhline(y=0)
plt.axvline(x=0)

# Show the grid on the plane
plt.grid()


# Draw a point on the plane
plt.plot(3, 2, marker="o")

plt.arrow(0, 0, -3, -2)
# show the window 
plt.show()