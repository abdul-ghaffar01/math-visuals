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
plt.plot(2.5, 5, marker="o")
plt.plot(5, 2.5, marker="o")
plt.plot(5, 5, marker="o")
plt.plot(2.5, 2.5, marker="o")


# Draw a line
x = [0, 5]
y = [0, 3]

plt.plot(x, y)

# Draw a triangle
x = [1, 5, 9, 1]
y = [1, 5, 1, 1]

plt.plot(x, y)

# Draw a square
square = [
    [-2.5, -2.5],
    [-2.5, -5],
    [-5, -5],
    [-5, -2.5],
    [-2.5, -2.5]
]

x = [p[0] for p in square]
y = [p[1] for p in square]

plt.plot(x, y)

# Draw multiple vectors
vectors = [
    [3, 5],
    [-3, 2],
    [4, -4]
]
for vector in vectors:
    plt.arrow(0, 0, vector[0], vector[1])


# ploting the arrow
plt.arrow(0, 0, -3, -2)


# Challenge #1
# Draw a mini house

house = [
    [-7.5, 5],
    [-5, 7.5],
    [-2.5, 5],
    [-7.5, 5],
    [-7.5, 0],
    [-2.5, 0],
    [-2.5, 5]
]

x = [p[0] for p in house]
y = [p[1] for p in house]

plt.plot(x, y)

# Challenge Completed

# show the window 
plt.show()