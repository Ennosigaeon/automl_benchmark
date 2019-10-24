from typing import List

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from adapter.base import BenchmarkResult
from benchmark import OpenML100Suite, OpenMLBenchmark
from evaluation.base import MongoPersistence


def plot_incumbent_performance(ls: List[BenchmarkResult], n: int = 250):
    benchmark = ls[0].benchmark
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    fig.set_dpi(250)

    solvers = {}
    for res in ls:
        for solver in res.solvers:
            x, y = solver.as_numpy(incumbent=True, x_axis='iterations')
            label = solver.algorithm
            if label == 'RoBo gp':
                label = 'RoBO'
            solvers.setdefault(label, []).append(y[:n])

    f_opt = benchmark.get_meta_information()['f_opt']
    ax.plot([1, len(next(iter(solvers.values()))[0]) + 1], [f_opt, f_opt], 'k', label='Optimum', linewidth=2)

    for name, values in solvers.items():
        y = np.vstack(values)
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)

        ax.fill_between(np.arange(1, n + 1, 1), mean - std, mean + std, alpha=0.25)
        ax.plot(np.arange(1, n + 1, 1), mean, label=name, linewidth=2)

    # Fixes for skewed plots
    if ls[0].name == 'Branin':
        ax.set_ylim(-10, 90)
    if ls[0].name == 'Rosenbrock10D':
        ax.set_ylim(-10, 270)
    if ls[0].name == 'Camelback':
        # ax.set_yscale('log')
        ax.set_ylim(0, 2000)

    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.set_title(ls[0].name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f(\\lambda)$')

    plt.savefig('evaluation/plots/{}_aggregated.pdf'.format(ls[0].name), bbox_inches='tight')
    # fig.show()
    # plt.show()


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


def plot_openml_100(persistence: MongoPersistence):
    tasks = OpenML100Suite.tasks()

    matplotlib.rcParams.update({'font.size': 12})

    ls = {
        'Random Search': [],
        'Grid Search': [],
        'SMAC': [],
        'hyperopt': [],
        'BOHB': [],
        'RoBO': [],
        'Optunity': [],
        'BTB': []
    }
    ids = []
    for id in tasks:

        benchmark = OpenMLBenchmark(id, load=False)

        d = {}
        results = persistence.load_all(benchmark)

        for res in results:
            valid = any([solver.score < 1 for solver in res.solvers])
            if valid:
                ids.append(id)

            for solver in res.solvers:

                y = solver.as_numpy()[1]

                key = solver.algorithm
                if key == 'RoBo gp':
                    key = 'RoBO'

                d.setdefault(key, []).append(y)

        for key, value in d.items():
            for list in value:
                if np.mean(list) < 1:
                    ls[key].append(list[0:325])

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.set_dpi(250)

    for key, value in ls.items():
        y = np.mean(np.vstack(value), axis=0)
        std = np.std(np.vstack(value), axis=0)
        print('{}: {:.4f} \\pm {:.4f}'.format(key, y[200], std[200]))

        ax.plot(np.arange(1, len(y) + 1, 1), y, label=key, linewidth=2.0)

    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Misclassification Rate')
    ax.legend(loc='upper right')
    plt.savefig('evaluation/plots/openml100.pdf', bbox_inches='tight')


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
