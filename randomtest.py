import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from random import randint as r

print(r(100, 999))


def x_gen(n: int = 5, x0: int = 0, step: int = 1) -> list[int]:
    x = []
    xi = x0
    for i in range(n):
        x.append(xi)
        xi = xi + step
    return x


def y_gen(n, a, b) -> list[int]:
    y = []
    for i in range(n):
        y.append(r(a, b))
    return y

n = r(3,99)
x = x_gen(n=n, x0=1, step=10)
y = y_gen(len(x), 0, 99)

sns.set_theme()
plt.plot(x, y)
sns.set_style("white")
plt.plot(x,y)
plt.show()

sns.load_dataset('mpg')
# type(cars)
print(y)