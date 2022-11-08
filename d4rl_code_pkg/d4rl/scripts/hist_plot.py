import argparse
import d4rl
import gym
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    #heatmap = np.sqrt(gaussian_filter(heatmap, sigma=s))
    heatmap = (gaussian_filter(heatmap, sigma=s))

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def plot():
    ENVS = ['antmaze-umaze-v0', 'antmaze-umaze-diverse-v0', 'antmaze-medium-play-v0', 'antmaze-medium-diverse-v0',
            'antmaze-large-play-v0', 'antmaze-large-diverse-v0', 'maze2d-umaze-v0', 'maze2d-medium-v0', 'maze2d-large-v0']
    for env_name in ENVS:
        env = gym.make(env_name)
        print(env.observation_space)
        dataset = env.get_dataset()

        observations = dataset['observations']
        rewards = dataset['rewards']
        actions = dataset['actions']

        if env_name.startswith('maze2d'):
            x = observations[:,0]
            y = observations[:,1]
        else:
            x = observations[:,1]
            y = observations[:,0]

        print(x.shape)
        print(y.shape)
        plt.figure()

        heatmap, extent = myplot(x, y, 2, bins=1000)

        cmap = cm.hot
        heatmap = cmap(heatmap)
        magnitudes = np.sum(heatmap[:,:,:3], axis=2) / 3
        heatmap[:,:,3] = magnitudes
        plt.imshow((heatmap), extent=extent, cmap=cmap) 

        plt.axis('off')
        plt.savefig(env_name+'.png', transparent=True, bbox_inches='tight', pad_inches=0)


def plot_lines():
    #ENVS = ['antmaze-umaze-v0', 'antmaze-umaze-diverse-v0', 'antmaze-medium-play-v0', 'antmaze-medium-diverse-v0',
    #        'antmaze-large-play-v0', 'antmaze-large-diverse-v0', 'maze2d-umaze-v0', 'maze2d-medium-v0', 'maze2d-large-v0']
    #ENVS = ['maze2d-umaze-v0', 'maze2d-medium-v0', 'maze2d-large-v0']

    #ENVS = ['antmaze-umaze-v0', 'antmaze-umaze-diverse-v0', 'antmaze-medium-play-v0', 'antmaze-medium-diverse-v0', 'antmaze-large-play-v0', 'antmaze-large-diverse-v0']
    ENVS = ['antmaze-large-play-v0', 'antmaze-large-diverse-v0']
    for env_name in ENVS:
        print(env_name)
        env = gym.make(env_name)
        dataset = env.get_dataset()

        N = 1000000
        observations = dataset['observations'][:N]
        rewards = dataset['rewards']
        actions = dataset['actions']
        terminals = dataset['terminals']

        xs = []
        ys = []
        obs_arr = []
        cntr = 0
        for i in range(observations.shape[0]-1):
            cntr += 1
            dist_term = np.linalg.norm(observations[i+1,:2] - observations[i,:2]) >= 3.0
            #print(terminals[i], np.linalg.norm(observations[i,:2] - prev_obs))
            #if terminals[i] or dist_term:
            if dist_term or cntr >= env._max_episode_steps:
                if len(obs_arr) > 0:
                    obs_arr = np.array(obs_arr)
                    if env_name.startswith('maze2d'):
                        x = obs_arr[:,0]
                        y = obs_arr[:,1]
                    else:
                        x = obs_arr[:,1]
                        y = obs_arr[:,0]
                    xs.append(x)
                    ys.append(y)
                obs_arr = []
                cntr = 0
            else:
                obs_arr.append(observations[i])

        #plt.figure(figsize=(640, 480))
        plt.figure()
        for (xarr, yarr) in zip(xs, ys):
            plt.plot(xarr, yarr, lw=0.1)
        plt.axis('off')

        plt.savefig(env_name+'.png', transparent=True, bbox_inches='tight', pad_inches=0)


def transp():
    def normal_pdf(x, mean, var):
        return np.exp(-(x - mean)**2 / (2*var))


    # Generate the space in which the blobs will live
    xmin, xmax, ymin, ymax = (0, 100, 0, 100)
    n_bins = 100
    xx = np.linspace(xmin, xmax, n_bins)
    yy = np.linspace(ymin, ymax, n_bins)

    # Generate the blobs. The range of the values is roughly -.0002 to .0002
    means_high = [20, 50]
    means_low = [50, 60]
    var = [150, 200]

    gauss_x_high = normal_pdf(xx, means_high[0], var[0])
    gauss_y_high = normal_pdf(yy, means_high[1], var[0])

    gauss_x_low = normal_pdf(xx, means_low[0], var[1])
    gauss_y_low = normal_pdf(yy, means_low[1], var[1])

    weights = (np.outer(gauss_y_high, gauss_x_high)
               - np.outer(gauss_y_low, gauss_x_low))

    # We'll also create a grey background into which the pixels will fade
    greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)

    # First we'll plot these blobs using ``imshow`` without transparency.
    vmax = np.abs(weights).max()
    imshow_kwargs = {
        'vmax': vmax,
        'vmin': -vmax,
        'cmap': 'RdYlBu',
        'extent': (xmin, xmax, ymin, ymax),
    }

    fig, ax = plt.subplots()
    ax.imshow(greys)
    ax.imshow(weights, **imshow_kwargs)
    ax.set_axis_off()

    # Create the figure and image
    # Note that the absolute values may be slightly different
    fig, ax = plt.subplots()
    ax.imshow(greys)
    ax.imshow(weights, alpha=alphas, **imshow_kwargs)
    ax.set_axis_off()
    plt.savefig('output.png', transparent=True, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    #env_name = 'antmaze-umaze-v0'
    #env_name = 'antmaze-medium-play-v0'
    plot_lines()
    #plot()


