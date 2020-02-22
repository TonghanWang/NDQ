import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.distributions as D
from scipy.stats import multivariate_normal


def draw_message_distributions(args, mu, sigma, g=None):
    if g:
        print(g.detach().cpu().numpy())
    if args.mac in ['cate_broadcast_comm_mac', 'cate_broadcast_comm_mac_full'] and args.comm_embed_dim == 2:
        save_dir = os.path.join(args.checkpoint_path.replace('models', 'plots'), str(args.loaded_model_ts)) + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        azim = 0.

        if save_dir.split('/')[-2] == 'dim_2_beta_1e-1':
            azim = 90.

        draw_message_distributions_tracker1_2d(mu, sigma, save_dir, azim)


def draw_message_distributions_tracker1_2d(mu, sigma, save_dir, azim):
    # sigma *= torch.where(sigma<0.01, torch.ones(sigma.shape).cuda(), 10*torch.ones(sigma.shape).cuda())
    mu = mu.view(-1, 2)
    sigma = sigma.view(-1, 2)

    s_mu = torch.Tensor([0.0, 0.0])
    s_sigma = torch.Tensor([1.0, 1.0])

    x = y = np.arange(100)
    t = np.meshgrid(x, y)

    d = D.Normal(s_mu, s_sigma)
    d1 = D.Normal(mu[0], sigma[0])
    d21 = D.Normal(mu[2], sigma[2])
    d22 = D.Normal(mu[3], sigma[3])
    d31 = D.Normal(mu[4], sigma[4])
    d32 = D.Normal(mu[5], sigma[5])

    print('Entropy')
    print(d1.entropy().detach().cpu().numpy())
    print(d21.entropy().detach().cpu().numpy())
    print(d22.entropy().detach().cpu().numpy())
    print(d31.entropy().detach().cpu().numpy())
    print(d32.entropy().detach().cpu().numpy())

    print('KL Divergence')

    for tt_i in range(3):
        d1 = D.Normal(mu[tt_i * 2 + 0], sigma[tt_i * 2 + 0])
        d2 = D.Normal(mu[tt_i * 2 + 1], sigma[tt_i * 2 + 1])
        print(tt_i,
              D.kl_divergence(d1, d2).sum().detach().cpu().numpy(),
              D.kl_divergence(d1,  d).sum().detach().cpu().numpy(),
              D.kl_divergence(d2,  d).sum().detach().cpu().numpy(),
              sigma[tt_i * 2 + 0].mean().detach().cpu().numpy(),
              sigma[tt_i * 2 + 1].mean().detach().cpu().numpy())

    # Numpy array of mu and sigma
    s_mu_ = s_mu.detach().cpu().numpy()
    mu_0 = mu[0].detach().cpu().numpy()
    mu_2 = mu[2].detach().cpu().numpy()
    mu_3 = mu[3].detach().cpu().numpy()
    mu_4 = mu[4].detach().cpu().numpy()
    mu_5 = mu[5].detach().cpu().numpy()

    s_sigma_ = s_sigma.detach().cpu().numpy()
    sigma_0 = sigma[0].detach().cpu().numpy()
    sigma_2 = sigma[2].detach().cpu().numpy()
    sigma_3 = sigma[3].detach().cpu().numpy()
    sigma_4 = sigma[4].detach().cpu().numpy()
    sigma_5 = sigma[5].detach().cpu().numpy()

    # Print
    print('mu and sigma')
    print(mu_0, sigma_0)
    print(mu_2, sigma_2)
    print(mu_3, sigma_3)
    print(mu_4, sigma_4)
    print(mu_5, sigma_5)

    # Create grid
    x = np.linspace(-5, 5, 5000)
    y = np.linspace(-5, 5, 5000)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal(s_mu_, [[s_sigma[0], 0], [0, s_sigma[1]]])

    # Agent 1
    # Create multivariate normal
    rv1 = multivariate_normal(mu_0, [[sigma_0[0], 0], [0, sigma_0[1]]])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos) + rv1.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('message')
    plt.tight_layout()
    plt.savefig(save_dir + 'agent1.png')
    ax.view_init(elev=0., azim=azim)
    plt.savefig(save_dir + ("agent1_0_%i.png" % int(azim)))
    ax.view_init(elev=90., azim=0.)
    plt.savefig(save_dir + "agent1_90_0.png")
    # plt.show()
    plt.close()

    # Agent 2
    # Create multivariate normal
    rv21 = multivariate_normal(mu_2, [[sigma_2[0], 0], [0, sigma_2[1]]])
    rv22 = multivariate_normal(mu_3, [[sigma_3[0], 0], [0, sigma_3[1]]])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos) + rv21.pdf(pos) + rv22.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('message')
    plt.tight_layout()
    plt.savefig(save_dir + 'agent2.png')
    ax.view_init(elev=0., azim=azim)
    plt.savefig(save_dir + ("agent2_0_%i.png" % int(azim)))
    ax.view_init(elev=90., azim=0.)
    plt.savefig(save_dir + "agent2_90_0.png")
    # plt.show()
    plt.close()

    # Agent 3
    rv31 = multivariate_normal(mu_4, [[sigma_4[0], 0], [0, sigma_4[1]]])
    rv32 = multivariate_normal(mu_5, [[sigma_5[0], 0], [0, sigma_5[1]]])

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos) + rv31.pdf(pos) + rv32.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('message')
    plt.tight_layout()
    plt.savefig(save_dir + 'agent3.png')
    ax.view_init(elev=0., azim=azim)
    plt.savefig(save_dir + ("agent3_0_%i.png" % int(azim)))
    ax.view_init(elev=90., azim=0.)
    plt.savefig(save_dir + "agent3_90_0.png")
    # plt.show()
    plt.close()

    # Overall
    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos) + rv1.pdf(pos) + rv21.pdf(pos) + rv22.pdf(pos) + rv31.pdf(pos) + rv32.pdf(pos),
                    cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('message')
    plt.tight_layout()
    plt.savefig(save_dir + 'overall.png')
    ax.view_init(elev=0., azim=azim)
    plt.savefig(save_dir + ("overall_0_%i.png" % int(azim)))
    ax.view_init(elev=90., azim=0.)
    plt.savefig(save_dir + "overall_90_0.png")
    # plt.show()
    plt.close()
