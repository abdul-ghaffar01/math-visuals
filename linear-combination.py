import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # Interactive mode

u = np.array([3, 2])
v = np.array([2, 3])

fig, ax = plt.subplots()

while True:
    a = int(input("a = "))
    b = int(input("b = "))
    au = a*u
    bv = b*v
    lc = au + bv

    ax.clear()

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

    ax.axhline(0)
    ax.axvline(0)

    ax.grid(True)
    ax.set_aspect('equal')

    # Plotting au
    ax.quiver(0, 0, au[0], au[1], angles='xy', scale_units='xy', scale=1, color='red' )
    plt.text(au[0], au[1], 'u')

    # plotting bv from the head of au
    ax.quiver(au[0], au[1], bv[0], bv[1], angles='xy', scale_units='xy', scale=1, color='blue' )
    plt.text((bv[0]+au[0]), (au[1] + bv[1]), 'v')

    # Plotting the linear combination
    ax.quiver(
        0, 0,
        lc[0], lc[1],
        angles='xy',
        scale_units='xy',
        scale=1,
        color='green'
    )
    plt.text(lc[0], lc[1]+1, 'lc')

    plt.draw()
    plt.pause(0.01)