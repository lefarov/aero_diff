import torch
import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from ..data import Panel, Mesh
from ..utilities import integral_normal, integral_tangential


def test_panel():
    r = 1.0
    theta = np.linspace(0, 2 * np.pi, 100)
    xc, yc = r * np.cos(theta), r * np.sin(theta)

    n_panels = 10
    theta_sample = np.linspace(0.0, 2 * np.pi, n_panels + 1)
    xs, ys = r * np.cos(theta_sample), r * np.sin(theta_sample)

    points = torch.tensor(np.vstack([xs, ys]), requires_grad=True).t()
    points.retain_grad()

    panels = []
    for i in range(n_panels):
        panels.append(Panel(points[i], points[i + 1]))

    # mesh = Mesh(points[:-1], points[1:])

    A = torch.eye(n_panels, dtype=torch.float64) * 0.5
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral_normal(panel_i, panel_j)

    b = torch.stack([-1.0 * torch.cos(panel.beta) for panel in panels])
    sigmas = torch.solve(b.unsqueeze(-1), A)[0]

    for panel, sigma in zip(panels, sigmas):
        panel.sigma = sigma

    A = torch.zeros((n_panels, n_panels), dtype=torch.float64)
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral_tangential(panel_i, panel_j)

    b = torch.stack([-1.0 * torch.sin(panel.beta) for panel in panels])
    vts = torch.matmul(A, sigmas.squeeze(-1)) + b

    for panel, vt in zip(panels, vts):
        panel.vt = vt

    for panel in panels:
        panel.cp = 1.0 - (panel.vt / 1.0) ** 2

    panels[7].cp.backward()

    cp_analytical = 1.0 - 4 * (yc / r) ** 2
    points_cp = np.asarray([pan.cp.detach().numpy() for pan in panels])
    points_c = np.asarray([pan.point_c.detach().numpy()[0] for pan in panels])

    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.xlabel('x', fontsize=16)
    plt.ylabel('$C_p$', fontsize=16)

    plt.plot(xc, cp_analytical, label='analytical', color='b', linestyle='-', linewidth=1, zorder=1)
    plt.scatter(points_c, points_cp, label='source-panel method', color='#CD2305', s=40, zorder=2)

    plt.title(f'Number of panels : {n_panels}', fontsize=16)
    plt.legend(loc='best', prop={'size': 16})
    plt.xlim(-1.0, 1.0)
    plt.ylim(-4.0, 2.0)

    # size = 6
    # plt.figure(figsize=(size, size))
    # plt.grid()
    # plt.xlabel('x', fontsize=16)
    # plt.ylabel('y', fontsize=16)
    # plt.plot(a, b, label='cylinder', color='b', linestyle='-', linewidth=1)
    # plt.plot(xs, ys, label='panels', color='#CD2305', linestyle='-', linewidth=2)
    #
    # points_a = np.asarray([pan.point_a.detach().numpy() for pan in panels])
    # points_c = np.asarray([pan.point_c.detach().numpy() for pan in panels])
    #
    # plt.scatter(points_a[:, 0], points_a[:, 1], label='end-points', color='#CD2305', s=40)
    # plt.scatter(points_c[:, 0], points_c[:, 1], label='center-points', color='k', s=40, zorder=3)
    # plt.legend(loc='best', prop={'size': 16})
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    plt.show()
