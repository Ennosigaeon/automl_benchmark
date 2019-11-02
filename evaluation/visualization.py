import itertools
import pickle
from typing import List, Dict

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

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
        if key == 'RoBo gp':
            key = 'RoBO'
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

        img = ax.scatter(x1, x2, c=diff, cmap='viridis', vmin=-0.5, vmax=0.5)
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


def plot_dataset_performance(values, labels: list, tasks: list, cash: bool = False):
    with open('assets/ds.pkl', 'rb') as f:
        datasets: Dict[int, Dataset] = pickle.load(f)

    ds_by_classes = {}
    ds_by_features = {}
    ds_by_instances = {}
    ds_by_missing = {}

    for ds in datasets.values():
        if ds.task_id not in tasks:
            continue

        idx = tasks.index(ds.task_id)

        ds_by_classes.setdefault(ds.NumberOfClasses, []).append(idx)
        ds_by_features.setdefault(ds.NumberOfSymbolicFeatures + ds.NumberOfNumericFeatures, []).append(idx)
        ds_by_instances.setdefault(ds.NumberOfInstances, []).append(idx)

        p = round(ds.NumberOfMissingValues / (
                ds.NumberOfInstances * (ds.NumberOfNumericFeatures + ds.NumberOfSymbolicFeatures)), 6)
        ds_by_missing.setdefault(p, []).append(idx)

    def plot(mapping: Dict[int, List[int]], title: str, max: int = None):
        tmp = sorted(mapping.keys())
        idx = tmp.index(max) if max is not None else len(tmp)

        x = np.array(tmp[:idx])
        y = []
        for v in x:
            tmp = values[mapping[v], :].mean(axis=0)
            y.append(tmp)
        y = np.array(y)

        fig, ax = plt.subplots()
        for i in range(len(labels)):
            ax.plot(x, y[:, i], label=labels[i])
        ax.legend()
        ax.set_title(title)
        fig.show()

    plot(ds_by_classes, 'Number of Classes', max=26)
    plot(ds_by_features, 'Number of Features', max=2001)
    plot(ds_by_instances, 'Number of Samples', max=14980)
    plot(ds_by_missing, 'Percentage Missing Values')


def plot_overall_performance(x, labels: list, cash: bool = False):
    # Create box plots
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.set_dpi(250)

    values = []
    for i in range(x.shape[1]):
        values.append(x[x[:, i] > 0, i])

    # notch shape box plot
    bplot = ax.boxplot(values,
                       notch=True,  # notch shape
                       vert=True,  # vertical box alignment
                       patch_artist=True,  # fill with color
                       labels=labels)  # will be used to label x-ticks

    if cash:
        ax.set_title('Performance of CASH Solvers')
    else:
        ax.set_title('Performance of AutoML Frameworks')

    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)

    ax.yaxis.grid(True)
    ax.set_ylabel('Misclassification Rate')

    if cash:
        ax.set_ylim([None, 2.0])
        plt.savefig('evaluation/plots/performance-cash.pdf', bbox_inches='tight')
    else:
        ax.set_ylim([None, 2.5])
        plt.savefig('evaluation/plots/performance-automl-framworks.pdf', bbox_inches='tight')


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
