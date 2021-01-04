import numpy as np
import pdb
import matplotlib.pyplot as plt

def bruggeman_lhs(eps_eff_mn, eps_eff_std, eps_medium):
    mn = (eps_eff_mn-eps_medium) / (eps_eff_mn+2*eps_medium)
    std = 3*eps_medium*eps_eff_std / (eps_eff_mn+2*eps_medium)**2
    return mn, std

def eps_inclusion(m_mn, m_std, eps_medium):
    mn = eps_medium*(1+2*m_mn) / (1 - m_mn)
    std = 3*eps_medium*m_std / (1-m_mn)**2
    return mn, std

# Results from SPM data using Bayesian inference and eps = 80
nk2m = np.array([27.9, 40.5, 56.1, 63.3])
nk2s = np.array([0.5, 0.4, 0.6, 0.8])
c = np.array([0.5, 1, 1.5, 2])

# Experimental results (1948)
eps_mn_map = {'eps_licl_mn':np.array([71.2, 64.2, 57, 51]),
              'eps_kcl_mn':np.array([73.5, 68.5, 63.5, 58.5]),
              'eps_nacl_mn':np.array([73.9, 69.1, 64.3, 59.0]),
              'eps_rbcl_mn':np.array([73.5, 68.5, 63.5, 58.5])}
eps_std_map = {'eps_licl_std':np.array([2, 2, 2, 2]),
               'eps_kcl_std':np.array([2, 2, 2, 2]),
               'eps_nacl_std':np.array([2, 2, 2, 2]),
               'eps_rbcl_std':np.array([2, 2, 2, 2])}

sigma_mn_map = {'sigma_licl_mn':np.array([81.0, 72.9, 67.0, 62.0]),
              'sigma_kcl_mn':np.array([117.2, 111.9, 107.6, 104.2]),
              'sigma_nacl_mn':np.array([85.6, 78.7, 73.0, 68.6]),
              'sigma_rbcl_mn':np.array([125.5, 114.7, 107.5, 104.5])}

eps_h2o = 80.0

name_map = {'licl':'LiCl',
            'kcl':'KCl',
            'nacl':'NaCl',
            'rbcl':'RbCl'}

salts = ['licl', 'kcl', 'nacl', 'rbcl']

fig, ax = plt.subplots(figsize=(8, 8))
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(3.0)
for axis in ['top', 'right']:
    ax.spines[axis].set_visible(False)


markers = ['o', 'v', 's']
color1 = "#2A00FB"
color2 = "#F18400"
color3 = "C2"
color4 = "C3"
cs = [color1, color2, color3]

fontsize = 24
ms = 12
capsize = 3.0
marker = 'o'
linewidth = 3.0
figsize = (7, 8)

for salt, color, marker in zip(salts, cs, markers):
    print(eps_mn_map['eps_{}_mn'.format(salt)])
    lhs_mn, lhs_std = bruggeman_lhs(eps_mn_map['eps_{}_mn'.format(salt)],
                                    eps_std_map['eps_{}_std'.format(salt)],
                                    eps_h2o)

    delta_mn = nk2m*0.01
    delta_std = nk2s*0.01

    ax.errorbar(delta_mn, lhs_mn, xerr=delta_std, yerr=lhs_std, label=name_map[salt],
                linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color
                )

    p, V = np.polyfit(delta_mn, lhs_mn, 1, cov=True)
    x = np.linspace(0.15, 0.7, 30)
    plt.plot(x, p[0]* x + p[1], color=color)

    print(name_map[salt])
    print("x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
    print("x_2: {} +/- {} \n".format(p[1], np.sqrt(V[1][1])))

    eps_i_mn, eps_i_std = eps_inclusion(p[0], np.sqrt(V[0][0]), eps_h2o)
    print("eps_inclusion: {} +/- {}".format(eps_i_mn, eps_i_std))

xticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=fontsize)
ax.set_xlim(0.0, 0.7)
yticks = [-0.2, -0.15, -0.1, -0.05, 0, 0.05]
ax.set_ylim(-0.2, 0.05)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=fontsize)

ax.set_ylabel(r'$\frac{\epsilon_{eff} - \epsilon_{m}}{\epsilon_{eff} + 2\epsilon_{m}}$', fontsize=fontsize)
ax.set_xlabel(r'$\delta_i$', fontsize=fontsize)

ax.legend(frameon=False, fontsize=fontsize - 4)
plt.tight_layout()
plt.savefig('bruggeman.png', dpi=400)

# Ionic conductivity fit

salts = ['licl', 'kcl', 'nacl', 'rbcl']

fig, ax = plt.subplots(figsize=(8, 8))
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(3.0)
for axis in ['top', 'right']:
    ax.spines[axis].set_visible(False)


markers = ['o', 'v', 's']
color1 = "#2A00FB"
color2 = "#F18400"
color3 = "C2"
color4 = "C3"
cs = [color1, color2, color3]

fontsize = 24
ms = 12
capsize = 3.0
marker = 'o'
linewidth = 3.0
figsize = (7, 8)

for salt, color, marker in zip(salts, cs, markers):
    print(sigma_mn_map['sigma_{}_mn'.format(salt)])
    s = sigma_mn_map['sigma_{}_mn'.format(salt)]
    delta_mn = nk2m * 0.01
    delta_std = nk2s * 0.01
    ax.errorbar(delta_mn, s, xerr=delta_std, label=name_map[salt],
                linestyle="", capsize=capsize, marker=marker, markersize=ms, color=color
                )

    p, V = np.polyfit(delta_mn, s, 1, cov=True)
    x = np.linspace(0.15, 0.7, 30)
    plt.plot(x, p[0]* x + p[1], color=color)

    print(name_map[salt])

    print("x_1: {} +/- {}".format(p[0], np.sqrt(V[0][0])))
    print("x_2: {} +/- {} \n".format(p[1], np.sqrt(V[1][1])))

    print("A sigma_1: {} +/- {}".format(p[1], np.sqrt(V[0][0])))
    print("A sigma_2: {} +/- {} \n".format(p[1] + p[0], np.sqrt(V[1][1] + V[0][0])))

xticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=fontsize)
ax.set_xlim(0.1, 0.7)
yticks = [50, 70, 90, 110, 130]
ax.set_ylim(50, 130)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=fontsize)

ax.set_ylabel(r'$\sigma_{eff}$', fontsize=fontsize)
ax.set_xlabel(r'$\delta_i$', fontsize=fontsize)

ax.legend(frameon=False, fontsize=fontsize - 4)
plt.tight_layout()
plt.savefig('conductivity.png', dpi=400)








