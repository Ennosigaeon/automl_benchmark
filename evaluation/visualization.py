import itertools
import pickle
from typing import List, Dict

import math
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import operator
from matplotlib import cm, patches

from adapter.base import BenchmarkResult
from evaluation.scripts import Dataset


def plot_evaluated_configurations(ls: List[BenchmarkResult]):
    matplotlib.rcParams.update({'font.size': 11})

    benchmark = ls[0].benchmark
    res = ls[0]

    bounds = benchmark.get_meta_information()['bounds']
    grid = []
    for bound in bounds:
        grid.append(np.linspace(bound[0], bound[1], 100))

    x = grid[0]
    y = grid[1]
    yy, xx = np.meshgrid(y, x)

    zz = np.zeros(xx.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            zz[i][j] = benchmark.objective_function([x[i], y[j]])['function_value']

    fig, ax = plt.subplots()

    # Create color mesh
    im = plt.pcolormesh(xx, yy, zz, norm=colors.LogNorm(vmin=zz.min(), vmax=zz.max()))
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Objective Function', rotation=90)

    c = ['r', 'k', 'silver']
    for idx, solver in enumerate([s for s in res.solvers if s.algorithm in ['Grid Search', 'Random Search', 'BOHB']]):
        x = np.zeros(len(solver.evaluations))
        y = np.zeros(len(solver.evaluations))
        for i in range(len(solver.evaluations)):
            x[i] = solver.evaluations[i].config['x0']
            y[i] = solver.evaluations[i].config['x1']

        label = solver.algorithm
        if label == 'BOHB':
            label = 'Guided Search'

        ax.scatter(x, y, s=10, c=c[idx], label=label)

    ax.legend(loc='upper left', framealpha=1, handletextpad=0)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    fig.tight_layout()

    plt.savefig('evaluation/plots/{}_tested_configurations.pdf'.format(ls[0].name), bbox_inches='tight')
    # fig.show()
    # plt.show()


def plot_method_overhead(ls: List[BenchmarkResult], line_plot: bool = True):
    matplotlib.rcParams.update({'font.size': 11})

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    fig.set_dpi(250)

    solvers = {}
    for solver in sum([res.solvers for res in ls], []):
        y = []
        previous = None
        for ev in solver.evaluations:
            if previous is not None:
                y.append((ev.start - previous) * 1000)
            previous = ev.end

        key = solver.algorithm
        solvers.setdefault(key, []).append(y[:494])

    if line_plot:
        def smooth(y, box_pts):
            n = len(y)
            a = np.hstack((np.tile(y[0], box_pts), y, np.tile(y[-1], box_pts)))
            box = np.ones(box_pts) / box_pts
            a_smooth = np.convolve(a, box, mode='same')
            return a_smooth[box_pts:n + box_pts]

        for name, values in solvers.items():
            y = np.mean(np.vstack(values), axis=0)[5:]
            x = np.arange(1, len(y) + 1, 1)

            ax.plot(x, smooth(y, 20), label=name, linewidth=2)

        ax.legend(loc='upper left')
        ax.set_xlabel('Iteration')
    else:
        labels = []
        values = []
        for name, overhead in solvers.items():
            y = np.vstack(overhead)
            values.append(np.mean(y, axis=0))
            labels.append(name)

        ax.boxplot(values, labels=labels, sym='')

    ax.set_title('Solver Overhead')
    ax.set_ylabel('Overhead in ms')

    plt.savefig('evaluation/plots/{}_overhead.pdf'.format(ls[0].name), bbox_inches='tight')


def plot_cash_incumbent(x, labels: list):
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.set_dpi(250)

    for idx in range(len(labels)):
        value = x[idx]
        mean = value.mean(axis=0)
        std = value.std(axis=0)
        ax.plot(np.arange(1, len(mean) + 1, 1), mean, label=labels[idx], linewidth=2.0)
        # ax.fill_between(np.arange(1, len(mean) + 1, 1), mean - std, mean + std, alpha=0.25)

    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Normalized Performance')
    ax.legend(loc='lower right')
    plt.savefig('evaluation/plots/cash-incumbent.pdf', bbox_inches='tight')


def plot_pairwise_performance(x, labels: list, cash: bool = False):
    indices = range(len(labels))
    for i, j in itertools.combinations(indices, 2):
        print(labels[i], '\t', labels[j])

        # Ignore failed data sets
        mask = np.logical_and(x[:, i] != 0, x[:, j] != 0)
        x1 = x[mask, i]
        x2 = x[mask, j]

        diff = abs(x1 - x2)

        fig, ax = plt.subplots()

        img = ax.scatter(x1, x2, c=diff, cmap='inferno', vmin=-0.5, vmax=0.5)
        # plt.colorbar(img)

        ax.set_axisbelow(True)
        ax.grid(zorder=-1000)

        ax.autoscale(False)
        ax.plot([-5, 5], [-5, 5], zorder=-1000, c='#b0b0b0')

        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])

        lower = min(np.min(x1), np.min(x2)) - 0.05
        upper = max(max(np.max(x1), np.max(x2)), 1.05)

        ax.set_xlim([lower, upper])
        ax.set_ylim([lower, upper])

        prefix = 'cash' if cash else 'frameworks'
        plt.savefig('evaluation/plots/comparison-{}-{}-{}.pdf'.format(prefix,
                                                                      labels[i].replace(' ', ''),
                                                                      labels[j].replace(' ', '')),
                    bbox_inches='tight')


def plot_dataset_performance(values, minimum, maximum, labels: list, tasks: list, rows: int = 20, cash: bool = False):
    with open('assets/ds.pkl', 'rb') as f:
        datasets: Dict[int, Dataset] = pickle.load(f)

    a4_size = (8.27, 11.69)

    normalized = []
    for i in range(len(tasks)):
        normalized.append([])
        for j in range(len(values)):
            normalized[-1].append([])

    # normalized = [[[]] * len(values)] * len(tasks)
    for idx, algo in enumerate(values):
        for idx2, ds in enumerate(algo):
            for sample in ds:
                if sample != 1:
                    # v = ((1 - sample) - minimum[idx2]) / (maximum[idx2] - minimum[idx2])
                    v = 1 - sample
                    normalized[idx2][idx].append(v)

    std = np.zeros(len(normalized))
    for idx in range(len(normalized)):
        start = 1 if cash else 0
        std[idx] = np.array([item for sublist in normalized[idx][start:] for item in sublist]).std()

    plot_idx = std.argsort()[-(rows * 2):]

    fig, axes = plt.subplots(1, 2, gridspec_kw={'wspace': 0.01, 'hspace': 0})
    fig.set_size_inches(a4_size)
    for i in range(2):
        axes[i].set_frame_on(False)
        axes[i].grid(True, linewidth=0.5, alpha=0.75)
        axes[i].set_axisbelow(True)
        axes[i].set_yticks(np.arange(rows))
        axes[i].tick_params(axis=u'both', which=u'both', length=0)
        axes[i].set_yticklabels([datasets[tasks[idx]].name[:15] for idx in plot_idx[i * rows: (i + 1) * rows]])
        axes[i].set_ylim([-0.5, rows - 0.5])
        axes[i].tick_params(axis='both', which='major', labelsize=6)
    axes[1].yaxis.tick_right()
    print([datasets[tasks[idx]].task_id for idx in plot_idx])

    for idx in range(len(values)):
        mean = [[], []]
        mean_y = [[], []]
        x = [[], []]
        y = [[], []]
        for i, idx2 in enumerate(plot_idx):
            idx3 = i % 2

            mean[idx3].append(np.array(normalized[idx2][idx]).mean())
            x[idx3] += normalized[idx2][idx]

            y_val = i // 2 + (len(values) / rows) - 0.05 - 0.1 * idx
            mean_y[idx3].append(y_val)
            y[idx3] += [y_val] * len(normalized[idx2][idx])

        for i in range(2):
            # noinspection PyProtectedMember
            color = next(axes[i]._get_lines.prop_cycler)
            label = None if i == 1 else labels[idx]
            axes[i].scatter(x[i], y[i], s=(matplotlib.rcParams['lines.markersize'] ** 2) * 0.33, alpha=0.33,
                            linewidths=0, **color)
            axes[i].scatter(mean[i], mean_y[i], marker='d', s=(matplotlib.rcParams['lines.markersize'] ** 2) * 0.5,
                            label=label, **color)

    handles, labels_txt = axes[0].get_legend_handles_labels()

    fig.subplots_adjust(bottom=0.033)
    fig.legend(handles, labels_txt, ncol=len(labels) // 2, loc='lower center', borderaxespad=0.1,
               fontsize=6)

    if cash:
        plt.savefig('evaluation/plots/performance-ds-cash.pdf', bbox_inches='tight')
    else:
        plt.savefig('evaluation/plots/performance-ds-frameworks.pdf', bbox_inches='tight')


def plot_overall_performance(x: List[List[List[float]]], labels: list, cash: bool = False):
    # Create box plots
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.set_dpi(75)

    zoom = 10
    values = [[item for sublist in l for item in sublist] for l in x]
    scaled_values = []
    for hpo in values:
        array = np.array(hpo)
        print(array.min(), array.max())

        top = array[array > 1.5]
        mid = array[(1.5 > array) & (array > 0.5)]
        bottom = array[array < 0.5]

        # Rescale values
        top = (top - 0.5) + (zoom - 1)
        mid = (mid - 0.5) * zoom
        bottom = bottom - 0.5

        # Remove failed configurations
        bottom = bottom[bottom > -5]

        scaled_values.append(np.hstack((top, mid, bottom)))

    bplot = ax.boxplot(scaled_values,
                       notch=True,
                       vert=True,
                       patch_artist=True,
                       labels=labels,
                       flierprops={'marker': 'x', 'alpha': 0.75, 'markerfacecolor': '#1f77b4',
                                   'markeredgecolor': '#1f77b4', 'markersize': 5})
    ax.autoscale(False)
    ax.plot([-10, 10], [0, 0], zorder=-1000, c='#b0b0b0')
    ax.plot([-10, 10], [zoom, zoom], zorder=-1000, c='#b0b0b0')

    ax.yaxis.grid(True)
    ax.set_ylabel('Normalized Performance')

    if cash:
        ax.set_title('Performance of CASH Solvers')
        ax.set_yticks([-2.5, -0.5, 0, zoom / 2, zoom] + [2 * i + 0.5 + zoom for i in range(0, 6)])
        ax.set_yticklabels([-2, 0, 0.5, 1, 1.5] + [2 * i for i in range(1, 7)])
        plt.savefig('evaluation/plots/performance-cash.pdf', bbox_inches='tight')
    else:
        ax.set_title('Performance of AutoML Frameworks')
        ax.set_yticks([-4.5, -2.5, -0.5, 0, zoom / 2, zoom] + [2 * i + 0.5 + zoom for i in range(0, 5)])
        ax.set_yticklabels([-4, -2, 0, 0.5, 1, 1.5] + [2 * i for i in range(1, 6)])
        plt.savefig('evaluation/plots/performance-automl-frameworks.pdf', bbox_inches='tight')


def plot_configuration_similarity(dicts: list, cash: bool = False):
    def flatten(l):
        return [item for sublist in l for item in sublist]

    algorithms = set()
    for dic in dicts:
        algorithms.update(flatten([i.keys() for i in dic[1].values()]))
    colours = iter(cm.jet(np.linspace(0, 1, len(algorithms))))
    colours = {k: v for k, v in zip(algorithms, colours)}
    base_size = (matplotlib.rcParams['lines.markersize'] ** 2) * 0.33

    def scatter(ax, dict, title):
        values = {}
        for task, d in dict.items():
            for algo, v in d.items():
                if algo not in values:
                    values[algo] = []
                values[algo] += v

        rect = patches.Rectangle((4.95, 0.725), 5.3, 0.325, edgecolor='lightgray', facecolor='lightgray', alpha=0.5)
        ax.add_patch(rect)

        for algo, array in values.items():
            tmp = np.array(array)
            x = tmp[:, 0]
            x[x > 10] = 10
            y = tmp[:, 1]
            s = tmp[:, 2] * base_size

            ax.scatter(x, y, label=algo.split('.')[-1], alpha=1, linewidths=0, s=s, c=[colours[algo]])

        ax.set_ylim([0, 1.05])
        ax.set_xlim([-0.25, 10.25])
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel('Instances per Cluster', fontsize=6, labelpad=1)
        ax.set_ylabel('Silhouette Coefficient', fontsize=6, labelpad=1)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='both', which='minor', labelsize=4)

    rows = max(1, math.ceil(len(dicts) / 3))
    fig, axes = plt.subplots(rows, 3 if len(dicts) > 1 else 1)

    if len(dicts) == 1:
        axes = [axes]
    axes = flatten(axes)

    handles = []
    labels = []
    for i, dict in enumerate(dicts):
        scatter(axes[i], dict[1], dict[0])
        h, l = axes[i].get_legend_handles_labels()
        for i in range(len(h)):
            try:
                labels.index(l[i])
            except ValueError:
                handles.append(h[i])
                labels.append(l[i])

    fig.delaxes(axes[-1])

    suffix = 'Classifier'
    labels = [s[:-len(suffix)] if s.endswith(suffix) else s for s in labels]
    for i in range(len(labels)):
        if labels[i] == 'LinearDiscriminantAnalysis':
            labels[i] = 'LDA'
        if labels[i] == 'QuadraticDiscriminantAnalysis':
            labels[i] = 'QDA'

    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles, labels = zip(*hl)
    leg = fig.legend(handles, labels, ncol=2, bbox_to_anchor=(0.995, 0.26), columnspacing=1, borderaxespad=0.1,
                     fontsize=6, scatterpoints=1, handletextpad=0.33)
    for handle in leg.legendHandles:
        handle.set_sizes([8.0])

    handles2 = []
    labels2 = []
    for i in np.arange(0.25, 2.1, 0.5):
        labels2.append(i)
        handles2.append(plt.scatter([], [], s=i * base_size, edgecolors='none', c='b'))
    fig.legend(handles2, labels2, ncol=len(labels2), bbox_to_anchor=(0.995, 0.30), columnspacing=1.2, borderaxespad=0.1,
               fontsize=6)
    fig.text(0.68, 0.31, 'Normalized Accuracy', fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99, bottom=0.06, top=0.98)

    if cash:
        plt.savefig('evaluation/plots/config-similarity-cash.pdf')
    else:
        plt.savefig('evaluation/plots/config-similarity-frameworks.pdf')


def plot_branin():
    from mpl_toolkits.mplot3d import Axes3D

    matplotlib.rcParams.update({'font.size': 12})

    # noinspection PyStatementEffect
    Axes3D.name

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    fig.set_dpi(250)
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, 15, 0.05)
    Y = np.arange(-5, 10, 0.05)
    X, Y = np.meshgrid(X, Y)

    Z = (Y - (5.1 / (4 * np.pi ** 2)) * X ** 2 + 5 * X / np.pi - 6) ** 2
    Z += 10 * (1 - 1 / (8 * np.pi)) * np.cos(X) + 10

    # noinspection PyUnresolvedReferences
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.01, edgecolors='k', antialiased=True)

    ax.view_init(30, 120)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([-5, 0, 5, 10])

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    ax.set_xlabel('$x_0$', rotation=0)
    ax.set_ylabel('$x_1$', rotation=0)
    ax.set_zlabel('f$(x_0,\, x_1)$', rotation=90)
    ax.set_title('Branin Function')

    plt.savefig('evaluation/plots/branin.pdf', bbox_inches='tight')


def plot_successive_halving():
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    fig.set_dpi(250)

    ax.plot([0, 0.125, 0.25, 0.5, 1], [1, 0.40, 0.27, 0.2, 0.16])

    ax.plot([0, 0.125, 0.25], [1, 0.65, 0.5])
    ax.plot([0, 0.125, 0.25], [1, 0.55, 0.40])

    ax.plot([0, 0.125, 0.25, 0.5], [1, 0.45, 0.25, 0.22])

    ax.plot([0, 0.125], [1, 0.9])
    ax.plot([0, 0.125], [1, 0.85])
    ax.plot([0, 0.125], [1, 0.76])
    ax.plot([0, 0.125], [1, 0.68])

    # Budget Constraints
    ax.plot([0.125, 0.125], [0, 1], c='k')
    ax.plot([0.25, 0.25], [0, 1], c='k')
    ax.plot([0.5, 0.5], [0, 1], c='k')
    ax.plot([1, 1], [0, 1], c='k')

    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Budget')
    ax.set_xticks([0.125, 0.25, 0.5, 1.0], ['12.5%', '25%', '50%', '100%'])

    plt.savefig('evaluation/plots/successive_halving.pdf', bbox_inches='tight')
