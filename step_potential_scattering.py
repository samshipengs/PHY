import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import animation
import time



t1 = time.time()
# some constants
beta = 4.
K0 = 1

upper_K = np.sqrt(2)
lower_K = 0.

def omega(K):
    return 1. - K**2 + 1j*K*np.sqrt(2 - K**2)

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return scipy.real(func(x))
    def imag_func(x):
        return scipy.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def integrand(u, tau):
    def df(K):
        return np.exp(-beta**2*(K-K0)**2)*np.exp(-1j*K**2*tau/2.)*(np.exp(1j*K*u)-np.exp(-1j*K*u)*omega(K))
    return df

def Psi(u_input, tau_input):
    return complex_quadrature(integrand(u_input, tau_input), lower_K, upper_K)[0]

def Psi_square(u_input, tau_input):
    Psi_result = Psi(u_input, tau_input)
    return scipy.real(np.conjugate(Psi_result)*Psi_result)


# In[4]:

tau_line = np.arange(-30, 10, 0.1)
u_line = np.arange(-30, 0, 0.1)


# In[13]:

result = []
for t in tau_line:
    result.append([Psi_square(i, t) for i in u_line])
print(len(result), len(result[0]))

t2 = time.time()
print('Total time took {0:.2f}min'.format((t2-t1)/60))
# In[16]:

fig = plt.figure()
ax = plt.axes(xlim=(-30, 0), ylim=(0, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,
# animation function.  This is called sequentially
def animate(t):
    x = u_line
    y = result[t]
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, range(len(result)), init_func=init,
                               interval=50, blit=True)
plt.show()


# In[ ]:



