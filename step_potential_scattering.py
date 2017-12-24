import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from helper import memoized


# use memoized in hope for speeding upo python integral calculation
# https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
exp = memoized(np.exp)
sqrt = memoized(np.sqrt)
conjugate = memoized(np.conjugate)


# some constants
beta = 4.
K0 = 1

upper_K = sqrt(2)
lower_K = 0.

@memoized
def omega(K):
    return 1. - K**2 + 1j*K*np.sqrt(2 - K**2)

@memoized
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

@memoized
def integrand(u, tau):
    def df(K):
        return exp(-beta**2*(K-K0)**2)*exp(-1j*K**2*tau/2.)*(exp(1j*K*u)-exp(-1j*K*u)*omega(K))
    return df

@memoized
def Psi(u_input, tau_input):
    return complex_quadrature(integrand(u_input, tau_input), lower_K, upper_K)[0]

@memoized
def Psi_square(u_input, tau_input):
    Psi_result = Psi(u_input, tau_input)
    return scipy.real(conjugate(Psi_result)*Psi_result)



# define range for tau and u
tau_line = np.arange(-30, 30, 1)
u_line = np.arange(-30, 0, 0.1)


# for each time point tau, we evaulate the integral over different u,
# save it to a list because we don't want to calculate it later on the fly for the animation
# since the computation takes a long time

t1 = time.time()
result = []
for t in tau_line:
    result.append([Psi_square(i, t) for i in u_line])
t2 = time.time()

print('Total time took {0:.2f}min'.format((t2-t1)/60))



# start creating the animation 
# see https://matplotlib.org/examples/animation/simple_anim.html 

fig = plt.figure()
ax = plt.axes(xlim=(-30, 0), ylim=(0, 1))
line, = ax.plot([], [], lw=2)


def animate(t):
    x = u_line
    y = result[t]
    line.set_data(x, y)
    return line,

# Init only required for blitting to give a clean slate.
def init():
    line.set_data([], [])
    return line,


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, range(len(result)), init_func=init,
                               interval=50, blit=True)

anim.save('barrier_E_lt_V_0.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()



