import matplotlib.pyplot as plt
import numpy as np

# consts
roll = 107

a = 1 + roll % 5
b = 1 + roll % 6
c = 1 + roll % 7
d = 1 + roll % 8
e = 1 + a % 5
f = 1 + b % 6

# eqn1
# x2 = (ab - bx1) / a
eq1_x1 = np.linspace(-10, 10, 100)
eq1_x2 = (a * b - b * eq1_x1) / a

# eqn2
# x2 = (-cd + dx1) / d
eq2_x1 = np.linspace(-10, 10, 100)
eq2_x2 = (-c * d + d * eq2_x1) / d

# eqn3
# x2 = (-ef - fx1) / e
eq3_x1 = np.linspace(-10, 10, 100)
eq3_x2 = (-e * f - f * eq3_x1) / e

plt.plot(eq1_x1, eq1_x2, 'r', label="bx1 + ax2 = ab")
plt.plot(eq2_x1, eq2_x2, 'g', label="dx1 - cx2 = cd")
plt.plot(eq3_x1, eq3_x2, 'b', label="fx1 + ex2 = -ef")
plt.legend()
plt.show()
