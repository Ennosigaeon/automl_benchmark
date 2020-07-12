import copy
import itertools
import operator
import pickle
from typing import List, Dict

import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.transforms as mtrans
from matplotlib import cm, patches
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.preprocessing import minmax_scale

import util
from evaluation.scripts import Dataset

FACE_COLOR = '#1f77b4'
AXIS_COLOR = '#b0b0b0'


def plot_cash_incumbent(x, x_std, labels: list):
    matplotlib.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    fig.set_dpi(100)

    axins = ax.inset_axes([0.66, 0.525, 0.325, 0.31])

    print('\t& '.join(['\\name{{{}}}'.format(labels[i]) for i in range(len(labels))]), end='\\\\\n')
    print('\t& '.join(['{:.4f}'.format(x_std[:, :, -1].mean(axis=1)[i]) for i in range(len(labels))]), end='\\\\\n')
    print('\t& '.join(['{:.4f}'.format(x[:, :, -1].std(axis=1)[i]) for i in range(len(labels))]), end='\\\\\n')

    for idx in range(len(labels)):
        mean = x[idx].mean(axis=0)
        x_tmp = np.arange(1, len(mean) + 1, 1)
        ax.plot(x_tmp, mean, label=labels[idx], linewidth=2.0)
        axins.plot(x_tmp, mean, label=labels[idx], linewidth=2.0)

    # sub region of the original image
    x1, x2, y1, y2 = 75, 325, 1.28, 1.34
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # axins.set_xscale('log')
    # axins.set_xticklabels([])
    axins.set_yticks([1.28, 1.3, 1.32, 1.34])

    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

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
        ax.plot([-5, 5], [-5, 5], zorder=-1000, c=AXIS_COLOR)

        ax.set_xlabel(labels[i], fontsize=12)
        ax.set_ylabel(labels[j], fontsize=12)

        ax.set_xlim([0.05, 1.05])
        ax.set_ylim([0.05, 1.05])

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
        axes[i].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        axes[i].tick_params(axis=u'both', which=u'both', length=0)
        axes[i].set_yticklabels([datasets[tasks[idx]].name[:15] for idx in plot_idx[i * rows: (i + 1) * rows]])
        axes[i].set_ylim([-0.5, rows - 0.5])
        axes[i].tick_params(axis='both', which='major', labelsize=8)
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

    fig.subplots_adjust(bottom=0.04)
    fig.legend(handles, labels_txt, ncol=len(labels) // 2, loc='lower center', borderaxespad=0.1,
               fontsize=8)

    # Get the bounding boxes of the axes including text decorations
    get_bbox = lambda ax: ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

    # Get the minimum and maximum extent, get the coordinate half-way between those
    xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).min()
    xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).max()
    xs = np.mean([xmax, xmin])

    ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).max()
    ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).min()

    line = plt.Line2D([xs, xs], [ymin, ymax], transform=fig.transFigure, color="black")
    fig.add_artist(line)

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
                       flierprops={'marker': 'x', 'alpha': 0.75, 'markerfacecolor': FACE_COLOR,
                                   'markeredgecolor': FACE_COLOR, 'markersize': 5})
    ax.autoscale(False)
    ax.plot([-10, 10], [0, 0], zorder=-1000, c=AXIS_COLOR)
    ax.plot([-10, 10], [zoom, zoom], zorder=-1000, c=AXIS_COLOR)

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


def plot_configuration_similarity(lists: List, cash: bool = False, bandwidth: float = None):
    algorithms = set()
    for dic in lists:
        algorithms.update(util.flatten([i.keys() for i in dic[1].values()]))
    base_size = (matplotlib.rcParams['lines.markersize'] ** 2) * 0.33

    def scatter(ax, dict, title):
        values = {}
        for task, d in dict.items():
            for algo, v in d.items():
                if algo not in values:
                    values[algo] = []
                values[algo] += v

        ax.add_patch(patches.Rectangle((4.95, 0.725), 5.3, 0.325, edgecolor='lightgray', facecolor='lightgray',
                                       alpha=0.5, zorder=-1000))
        for algo, array in values.items():
            tmp = np.array(array)
            x = tmp[:, 0]
            x[x > 10] = 10
            y = tmp[:, 1]
            s = tmp[:, 2] * base_size

            ax.scatter(x, y, label=algo.split('.')[-1], alpha=0.5, linewidths=0, s=s, c=FACE_COLOR)

        ax.set_ylim([-0.1, 1.05])
        ax.set_xlim([-0.25, 10.25])
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel('Instances per Cluster', fontsize=6, labelpad=1)
        ax.set_ylabel('Silhouette Coefficient', fontsize=6, labelpad=1)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='both', which='minor', labelsize=4)

    rows = max(1, math.ceil(len(lists) / 3))
    fig, axes = plt.subplots(rows, 3 if len(lists) > 1 else 1)

    if len(lists) == 1:
        axes = [axes]
    axes = util.flatten(axes)

    for i, d in enumerate(lists):
        # Skip 'All' view
        if i == 0:
            continue
        scatter(axes[i - 1], d[1], d[0])
    fig.delaxes(axes[-1])
    handles = []
    labels = []
    for i in np.arange(0.5, 6.1, 0.5):
        labels.append(i)
        handles.append(plt.scatter([], [], s=i * base_size, edgecolors='none', c=FACE_COLOR))

    fig.text(0.72, 0.3, 'Normalized Performance', fontsize=6)
    fig.legend(handles, labels, ncol=2, borderaxespad=0.05, bbox_to_anchor=(0.87, 0.29), fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.99, bottom=0.06, top=0.98)
    # fig.show()

    if cash:
        plt.savefig('evaluation/plots/config-similarity-cash-{}.pdf'.format(bandwidth))
    else:
        plt.savefig('evaluation/plots/config-similarity-frameworks.pdf')


def plot_pipeline_similarity(G: nx.Graph, seed: int = 8):
    labels_mapping = {}
    labels = {}

    node_frequency = []
    node_list = []
    color_map = []
    H = nx.bfs_tree(G, '__root__')
    for n in H.nodes:
        node_list.append(n)
        color_map.append(FACE_COLOR)
        node_frequency.append(G.nodes[n]['count'])

        label = G.nodes[n]['label']
        if len(label) > 0:
            if label not in labels_mapping:
                labels_mapping[label] = len(labels_mapping) + 1
            labels[n] = labels_mapping[label]

    node_frequency = np.array(node_frequency)
    # Scale most prominent nodes down
    node_frequency[node_frequency > 250] = 250
    color_map[node_list.index('__root__')] = 'r'

    node_frequency = (node_frequency / np.sum(node_frequency))
    node_frequency = node_frequency / node_frequency.max()

    edge_frequency = []
    edges = []
    for e in H.edges:
        edges.append(e)
        edge_frequency.append(G.edges[e]['count'])
    edge_frequency = np.array(edge_frequency)
    # Scale most prominent edges down
    edge_frequency[edge_frequency > 250] = 250

    edge_frequency = (edge_frequency / np.sum(edge_frequency))
    edge_frequency = edge_frequency / edge_frequency.max()

    fig, ax = plt.subplots()
    fig.set_size_inches((15, 12))

    # Stretch points over complete axis
    pos = nx.nx_agraph.graphviz_layout(H, prog='neato', args='-sep=.1 -Gepsilon=.00001 -Gstart={}'.format(seed))
    tmp = minmax_scale(np.asarray([pos[v] for v in node_list]))

    # Stretch points in central cluster
    center = tmp[0, :]
    for i in range(1, len(node_list)):
        p = tmp[i, :] - center
        if abs(p[0]) < 0.05:
            tmp[i, 0] = center[0] + 2 * p[0]
            tmp[i, 1] = center[1] + 1.5 * p[1]

    # Make space for legend
    y_offset = 0.135
    for i in range(len(node_list)):
        pos[node_list[i]] = (tmp[i, 0], (tmp[i, 1]) * (1 - y_offset) + y_offset)

    nx.draw(H, pos=pos, ax=ax, node_list=node_list, node_color=color_map, node_size=node_frequency * 300,
            edge_list=edges, edge_color=edge_frequency, edge_cmap=plt.get_cmap('binary'), edge_vmin=-0.1, arrowsize=5,
            labels=labels, font_size=12)

    handles = []
    labels = []
    for label, id in sorted(labels_mapping.items(), key=operator.itemgetter(1)):
        handles.append(ax.text(x=0, y=0, s=id, c='w', fontsize=1))

        if label.endswith('Classifier'):
            labels.append(label[:-len('Classifier')])
        else:
            labels.append(label)

    class TextHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            h = copy.copy(orig_handle)
            h.set_color('k')
            h.set_position((width / 2., height / 2.))
            h.set_transform(trans)
            h.set_ha("center")
            h.set_va("center")
            fp = orig_handle.get_font_properties().copy()
            fp.set_size(fontsize)
            # uncomment the following line,
            # if legend symbol should have the same size as in the plot
            h.set_font_properties(fp)
            return [h]

    fig.legend(handles, labels, ncol=6, loc='lower center', borderaxespad=0.1, fontsize=11,
               handler_map={type(handles[0]): TextHandler()})

    ax.set_ylim([-0.001, 1.005])
    ax.set_xlim([-0.001, 1.005])

    fig.show()
    fig.savefig('evaluation/plots/pipelines.pdf', bbox_inches='tight')


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
