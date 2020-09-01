import torch


def integral_normal(panel_i, panel_j, n=100):
    s = torch.linspace(0, panel_j.length.detach().numpy().item(), n)
    nom = ((panel_i.point_c[0] - (panel_j.point_a[0] - torch.sin(panel_j.beta) * s)) * torch.cos(panel_i.beta) +
           (panel_i.point_c[1] - (panel_j.point_a[1] + torch.cos(panel_j.beta) * s)) * torch.sin(panel_i.beta))

    denom = ((panel_i.point_c[0] - (panel_j.point_a[0] - torch.sin(panel_j.beta) * s)) ** 2 +
             (panel_i.point_c[1] - (panel_j.point_a[1] + torch.cos(panel_j.beta) * s)) ** 2)

    return torch.trapz(nom / denom, s)


# def integral_norm_batch(points_c, points_a, beta, n=100):


def integral_tangential(panel_i, panel_j, n=100):
    s = torch.linspace(0, panel_j.length.detach().numpy().item(), n)
    nom = (- (panel_i.point_c[0] - (panel_j.point_a[0] - torch.sin(panel_j.beta) * s)) * torch.sin(panel_i.beta) +
           (panel_i.point_c[1] - (panel_j.point_a[1] + torch.cos(panel_j.beta) * s)) * torch.cos(panel_i.beta))

    denom = ((panel_i.point_c[0] - (panel_j.point_a[0] - torch.sin(panel_j.beta) * s)) ** 2 +
             (panel_i.point_c[1] - (panel_j.point_a[1] + torch.cos(panel_j.beta) * s)) ** 2)

    return torch.trapz(nom / denom, s)
