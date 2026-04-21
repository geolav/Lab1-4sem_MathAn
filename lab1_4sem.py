import math

import numpy as np
import matplotlib.pyplot as plt
import time

from matplotlib.widgets import Slider

# Values from Analytic method:
lebegue_exact = 4 * np.log(4) - 3
ls_exact = sum(np.log(math.sqrt(k)) for k in range(1, 16))

print("=" * 60)
print("АНАЛИТИЧЕСКИЕ ЗНАЧЕНИЯ")
print(f"Интеграл Лебега:           {lebegue_exact:.10f}")
print(f"Интеграл Лебега-Стилтьеса: {ls_exact:.10f}")
print("=" * 60)


# functions
def f(x):
    return np.log(x)

def f_simple(x, n):
    if n == 0:
        return np.zeros_like(x)
    return np.floor(n * np.log(x)) / n



# 2.1 plot f_n graphic with "n" parameter slider

x = np.linspace(1, 4, 1000)
n0 = 10
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
line_f, = ax.plot(x, f(x), label='ln(x)', linewidth=2)
line_fn, = ax.step(x, f_simple(x, n0), where='post', label=f'f_n (n={n0})')

ax.set_xlim(1, 4)
ax.set_ylim(-0.1, 1.5)
ax.set_title(f'Аппроксимация ln(x), n = {n0}')
ax.legend()
ax.grid()

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(
    ax=ax_slider,
    label='n',
    valmin=1,
    valmax=400,
    valinit=n0,
    valstep=1
)

def update(val):
    n = int(slider.val)
    line_fn.set_ydata(f_simple(x, n))
    line_fn.set_label(f'f_n (n={n})')
    ax.set_title(f'Аппроксимация ln(x), n = {n}')
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()



# 2.1 plot f_n graphic

x = np.linspace(1, 4, 10000)
plt.figure(figsize=(10, 6))
n_min, n_max = 1, 400
for n in [1, 2, 5, 10, 50, 100, 500]:
    norm = (np.log(n) - np.log(n_min)) / (np.log(n_max) - np.log(n_min))
    color = plt.cm.viridis(0.3 + 0.7 * norm)
    linewidth = 3 / (1 + np.log2(n))
    plt.step(x, f_simple(x, n), where='post', linewidth=linewidth, label=f'f_{n}')

plt.plot(x, f(x), 'k', linewidth=2, label='ln(x)')
plt.legend()
plt.title('Аппроксимация f_n → ln(x)')
plt.grid()
plt.savefig("graph_fn.png", dpi=150)
plt.show()


# 2.2 Lebegue integral calculation

def lebegue_integral_fn(n):
    result = 0.0
    i_max = int(np.floor(n * np.log(4)))
    for i in range(i_max + 1):
        left = max(np.exp(i / n), 1.0)
        right = min(np.exp((i + 1) / n), 4.0)
        if right > left:
            result += (i / n) * (right - left)
    return result


print("\nИнтеграл Лебега ∫(f_n)dλ")
print(f"{'n':>6} {'значение':>15} {'ошибка':>15} {'время':>10}")

for n in [10, 100, 1000, 10000, 50000]:
    t0 = time.perf_counter()
    val = lebegue_integral_fn(n)
    t = time.perf_counter() - t0
    err = abs(val - lebegue_exact)
    print(f"{n:>6} {val:>15.10f} {err:>15.2e} {t:>10.6f}")


# 2.2 Lebegue-Stilties integral calculation

def ls_integral_fn(n):
    result = 0.0
    # jump points (at which the Lebesgue-Stilties measure is concentrated) sqrt(k), k = 1..16
    for k in range(1, 16):
        x0 = np.sqrt(k)
        result += f_simple(x0, n)
    return result


print("\nИнтеграл Лебега-Стилтьеса ∫(f_n)dμ_F")
print(f"{'n':>6} {'значение':>15} {'ошибка':>15} {'время':>10}")

for n in [10, 100, 1000, 10000, 100000, 500000]:
    t0 = time.perf_counter()
    val = ls_integral_fn(n)
    t = time.perf_counter() - t0
    err = abs(val - ls_exact)
    print(f"{n:>6} {val:>15.10f} {err:>15.2e} {t:>10.6f}")