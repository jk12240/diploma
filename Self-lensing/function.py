import matplotlib
matplotlib.use('Agg')  # Используем 'Agg' бэкэнд для отрисовки в памяти
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
import astropy.units as u
from matplotlib.cm import ScalarMappable


G = const.G.value
c = const.c.value
M_sun = const.M_sun.value

cosmo = FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc, Om0=0.3)


def plot_grav_lens_jet(z_S, M_in, D_LS, length, phi, x_c, y_c, n):
  z_S = float(z_S)
  M = float(M_in)
  length = float(length)
  phi = float(phi)
  x_c = float(x_c)
  y_c = float(y_c)
  n = int(n)
  D_LS = (float(D_LS) * u.pc).to(u.m).value
  D_S = cosmo.angular_diameter_distance(z_S).to(u.m).value
  D_L = D_S - D_LS

  M = M_sun * M * 10 ** 7

  phi = np.deg2rad(phi)
  Theta_E = np.sqrt(4 * G * M / c ** 2 * D_LS / (D_L * D_S))
  Theta_E = (Theta_E * u.radian).to(u.mas).value
  start_x = float(x_c - length/2 * np.cos(phi))
  end_x = float(x_c + length/2 * np.cos(phi))
  start_y = float(y_c - length/2 * np.sin(phi))
  end_y = float(y_c + length/2 * np.sin(phi))


  a = np.linspace(start_x, end_x, num=n)
  b = np.linspace(start_y, end_y, num=n)

  k = []
  x1 = []
  x2 = []
  y = []
  mu1 = []
  mu2 = []

  for i in range(len(a)):
    y.append(np.sqrt(a[i] ** 2 + b[i] ** 2))

  for i in range(len(a)):
    k.append(b[i] / a[i])
    x1.append(y[i] / 2 - np.sqrt((y[i] / 2) ** 2 + 1))
    x2.append(y[i] / 2 + np.sqrt((y[i] / 2) ** 2 + 1))

  for i in range(len(a)):
    mu1.append(abs((1 - (1 / x1[i]) ** 4) ** (-1)))
    mu2.append(abs((1 - (1 / x2[i]) ** 4) ** (-1)))

  fig = plt.figure(figsize=[16, 9])
  ax = fig.gca()
  fig = plt.gcf()

  plt.plot(0, 0, '.', color='k', markersize=10) 

  circle = plt.Circle((0, 0), 1 * Theta_E, fill=False)
  ax.add_patch(circle) # plotting the Einstein radius

  mu = np.concatenate((np.array(mu1), np.array(mu2))) 

  for i in range(0, len(y), int(n/25)): 
    plt.scatter(a[i] * Theta_E, b[i] * Theta_E, color='gray', s=20)

  plt.scatter([], [], color='gray', s=20, label='Jet')

  for i in range(len(y)):
    if a[i] < 0:
      plt.scatter(-x1[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), -x1[i] * k[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), c=mu1[i], cmap='plasma', vmin=min(mu), vmax=max(mu), s=20*mu1[i])
      plt.scatter(-x2[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), -x2[i] * k[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), c=mu2[i], cmap='plasma', vmin=min(mu), vmax=max(mu), s=20*mu2[i])
    if a[i] >= 0:
      plt.scatter(x1[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), x1[i] * k[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), c=mu1[i], cmap='plasma', vmin=min(mu), vmax=max(mu), s=20*mu1[i])
      plt.scatter(x2[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), x2[i] * k[i] * Theta_E / np.sqrt(k[i] ** 2 + 1), c=mu2[i], cmap='plasma', vmin=min(mu), vmax=max(mu), s=20*mu2[i])

  sm = ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(mu), vmax=max(mu)))
  sm.set_array([])
  plt.colorbar(sm, label='Magnification')

  plt.axis('equal')
  plt.xlabel('x, mas')
  plt.ylabel('y, mas')
  D_LS_pc = (D_LS * u.m).to(u.pc).value
  plt.title(f'z = {z_S}, M = {M_in} * 10^7 * M_sun, D_LS = {D_LS_pc} pc, length = {length}, phi = {round(np.rad2deg(phi))} deg, x_c = {x_c}, y_c = {y_c}, n = {n}')
  plt.grid()
  plt.legend()
  return fig 

def save_image(fig):
    import io
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png')
    img_io.seek(0)
    return img_io
