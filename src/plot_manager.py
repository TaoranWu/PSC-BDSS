import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Polygon
from src.utils.interval import Interval
from src.utils.ellipsoid import Ellipsoid
from sympy import symbols, lambdify, sympify
from src.template import Templates
import datetime
import random


class PlotManager:
    def __init__(self,
                 dim,
                 project_values,
                 safe_set=None,
                 K=0,
                 width: int = 800, height: int = 800,
                 grid: bool = True,
                 axis_equal: bool = True,
                 show_color_bar: bool = True,
                 v_filled: bool = False,
                 safe_set_filled: bool = False,
                 safe_set_color="red",
                 save=False,
                 prob_name="",
                 is_iteration=False,
                 save_file="pdf"):

        px = 1 / plt.rcParams["figure.dpi"]
        self.fig_list, self.ax_list = [], []
        for i in range(len(dim)):
            # fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")
            fig, ax = plt.subplots(figsize=(width * px, height * px))
            self.fig_list.append(fig)
            self.ax_list.append(ax)
        self.dim = dim
        self.num_dim = len(self.dim)
        self.project_values = project_values

        self.safe_set = safe_set
        self.v_filled = v_filled
        self.show_color_bar = show_color_bar
        self.safe_set_filled = safe_set_filled
        self.safe_set_color = safe_set_color
        self.grid = grid
        self.axis_equal = axis_equal
        self.is_iteration = is_iteration
        self.iteration = 0
        self.K = K
        self.num_v_split = 500
        self.save = save
        self.prob_name = prob_name
        self.save_file = save_file

        if is_iteration:
            # self.cmap = get_cmap('viridis')
            # self.cmap = [[0.6, 0.7, 0.8], [0.5, 0.65, 0.75], [0.4, 0.6, 0.7], [0.3, 0.55, 0.65], [0.2, 0.5, 0.6], [0.1, 0.45, 0.55], [0.0, 0.4, 0.5], [0.0, 0.35, 0.45], [0.0, 0.3, 0.4], [0.0, 0.25, 0.35], [0.0, 0.2, 0.3], [0.0, 0.15, 0.25], [0.0, 0.1, 0.2], [0.0, 0.05, 0.15], [0.0, 0.03, 0.1], [0.0, 0.02, 0.05], [0.0, 0.01, 0.03]]
            self.cmap = [[0.6, 0.8, 1.0], [0.4, 0.7, 1.0], [0.2, 0.6, 0.9],
                         [0.1, 0.5, 0.8], [0.0, 0.4, 0.7], [0.0, 0.2, 0.6]]

    def set_safe_set(self, safe_set):
        self.safe_set = safe_set

    def show(self):
        for i in range(len(self.ax_list)):
            ax = self.ax_list[i]
            if self.grid:
                ax.grid()
            if self.axis_equal:
                ax.axis('equal')

            # ax.set_xlim(self.safe_set.inf[self.dim[i][0]] - 0.1 * abs(self.safe_set.inf[self.dim[i][0]]),
            #             self.safe_set.sup[self.dim[i][0]] + 0.1 * abs(self.safe_set.sup[self.dim[i][0]]))
            # ax.set_ylim(self.safe_set.inf[self.dim[i][1]] - 0.1 * abs(self.safe_set.inf[self.dim[i][1]]),
            #             self.safe_set.sup[self.dim[i][1]] + 0.1 * abs(self.safe_set.sup[self.dim[i][1]]))

            # ax.set_xlim([-1.1, 1.1])
            # ax.set_ylim([-1.1, 1.1])
            ax.set_xlim([-0.85, 0.85])
            ax.set_ylim([-0.85, 0.85])

            # ax.set_xlabel(r'$\theta$', fontsize=14)
            # ax.set_ylabel(r'$\dot{\theta}$', fontsize=14)

            ax.set_xlabel(r'$x$', fontsize=14)
            ax.set_ylabel(r'$y$', fontsize=14)
        if self.save:
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            folder_path = '../../fig_results/'
            # folder_path = '../fig_results/'
            if self.save_file == "pdf":
                file_path = f'{folder_path}/{self.prob_name}_{current_time}.pdf'
            else:
                file_path = f'{folder_path}/{self.prob_name}_{current_time}.jpg'
            plt.savefig(file_path, dpi=1000, transparent=True, bbox_inches='tight')
        plt.show()

    def add_safe_set(self):
        if isinstance(self.safe_set, Interval):
            for i in range(self.num_dim):
                self.ax_list[i].add_patch(
                    Polygon(
                        self.safe_set.proj(self.dim[i]).rectangle(),
                        closed=True,
                        alpha=1,
                        fill=self.safe_set_filled,
                        linewidth=3,
                        edgecolor=self.safe_set_color,
                        facecolor=self.safe_set_color,
                        zorder=101,
                    )
                )
        elif isinstance(self.safe_set, Ellipsoid):
            for i in range(self.num_dim):
                theta = np.linspace(0, 2 * np.pi, 100)
                ellipse_x = (self.safe_set.center[self.dim[i][0]] +
                             1 / np.sqrt(self.safe_set.coefficients[self.dim[i][0]]) * np.cos(theta))
                ellipse_y = (self.safe_set.center[self.dim[i][1]] +
                             1 / np.sqrt(self.safe_set.coefficients[self.dim[i][1]]) * np.sin(theta))
                self.ax_list[i].plot(ellipse_x, ellipse_y, label="Ellipsoid Boundary", linewidth=5,
                                     color=self.safe_set_color)
        else:
            raise NotImplementedError()

    def add_traj(self, points_to_scatter, color="black"):
        assert self.safe_set.degree == 2
        x_values, y_values = zip(*points_to_scatter)
        if color == "black":
            c = "black"
        elif color == "random":
            # c = np.random.rand(3, )
            c = random.choice(list(mcolors.TABLEAU_COLORS.values()))
        else:
            c = color
        self.ax_list[0].scatter(x_values, y_values, color=c, s=50, marker='o', zorder=100)
        self.ax_list[0].plot(x_values, y_values, color=c, linestyle='--', linewidth=3)
        self.ax_list[0].scatter(x_values[0], y_values[0], color='black', s=400, marker='*', zorder=101)

    def add_monte_carlo(self, sim_data):
        # assert self.safe_set.degree == 2
        # plt.scatter(sim_data[:, 0], sim_data[:, 1], s=0.1, color="gray", marker='o')
        self.ax_list[0].scatter(sim_data[:, 0], sim_data[:, 1], s=0.1, color=[(0.7, 0.7, 0.7)], marker='o')

        # x = sim_data[:, 0]
        # y = sim_data[:, 1]
        # from scipy.stats import kde
        # data = np.vstack([x, y])
        # k = kde.gaussian_kde(data)
        # xi, yi = np.mgrid[x.min():x.max():20j, y.min():y.max():20j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        #
        # plt.contourf(xi, yi, zi.reshape(xi.shape), levels=1, color=["gray"])

    def add_v(self, v_str: str = None):
        vars_num = self.safe_set.degree
        for i in range(self.num_dim):
            all_symbols = symbols('x0:{}'.format(vars_num))
            v_sym = sympify(v_str)

            projected_vars = [all_symbols[i] for i in self.dim[i]]
            if self.project_values != {}:
                for dim, val in self.project_values[i].items():
                    v_sym = v_sym.subs(all_symbols[dim], val)
            v_lambdified = lambdify(projected_vars, v_sym, modules='numpy')
            x_vals = np.linspace(self.safe_set.inf[self.dim[i][0]], self.safe_set.sup[self.dim[i][0]], self.num_v_split)
            y_vals = np.linspace(self.safe_set.inf[self.dim[i][1]], self.safe_set.sup[self.dim[i][1]], self.num_v_split)
            x, y = np.meshgrid(x_vals, y_vals)

            all_vals = np.zeros((x.shape[0], x.shape[1], vars_num))
            all_vals[..., self.dim[i][0]] = x
            all_vals[..., self.dim[i][1]] = y

            vf = v_lambdified(*[all_vals[..., dim] for dim in self.dim[i]])

            if self.safe_set.degree == 2:
                reshaped_x = x.reshape(self.num_v_split ** 2, 1)
                reshaped_y = y.reshape(self.num_v_split ** 2, 1)
                x_data = np.hstack((reshaped_x, reshaped_y))
                is_in_safe_set = []
                if isinstance(self.safe_set, Interval):
                    is_in_safe_set = np.all((x_data > self.safe_set.inf) & (x_data < self.safe_set.sup), axis=1)
                elif isinstance(self.safe_set, Ellipsoid):
                    is_in_safe_set = self.safe_set.is_in_safe_set(x_data)
                is_in_safe_set = is_in_safe_set.reshape(self.num_v_split, self.num_v_split)
                vf = np.where(is_in_safe_set, vf, -1)

            if self.is_iteration and self.K != 0:
                # color = self.cmap(float(10 - 1 - self.iteration) / 10)
                if self.iteration < len(self.cmap):
                    color = self.cmap[self.iteration]
                else:
                    color = self.cmap[-1]
                # color = (0, 0, 1)
            else:
                color = (0.5, 0.7, 0.95)
                # color = (0, 0, 1)

            if self.v_filled:
                # self.ax_list[i].contourf(x, y, vf, levels=np.linspace(vf.min(), 0, 100), colors='white', alpha=0)
                self.ax_list[i].contourf(x, y, vf, levels=[0, vf.max()], colors=[color], alpha=0.7)
                self.ax_list[i].contour(x, y, vf, levels=[0], colors=[color], linewidths=3)
                # self.ax_list[i].contourf(x, y, vf, levels=np.linspace(vf.min(), vf.max(), 100), cmap='RdYlBu', alpha=0.5)
            else:
                self.ax_list[i].contour(x, y, vf, levels=[0], colors=[color], linewidths=6)

            # if self.v_filled and self.show_color_bar:
            #     self.ax_list[i].colorbar(label='f(x0, x1)')

        self.iteration += 1

    @staticmethod
    def surface_3d_plot(coe, num, lb, ub, template: Templates, xlim=None, ylim=None):
        assert lb.shape[0] == 2
        x = np.linspace(lb[0], ub[0], num)
        y = np.linspace(lb[1], ub[1], num)

        X, Y = np.meshgrid(x, y)
        xy_flat = np.column_stack([X.ravel(), Y.ravel()])

        Z_flat = template.calc_values(xy_flat)
        Z_flat = np.dot(Z_flat, coe)

        Z = Z_flat.reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            edgecolor='none',
            linewidth=0,
            antialiased=True,
            shade=True
        )

        ax.view_init(elev=30, azim=-45)


        ax.set_aspect('equal')
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_xlabel(r'$x$', fontsize=12)
        ax.set_ylabel(r'$y$', fontsize=12)
        ax.set_zlabel(r'$h$', fontsize=12)
        plt.tight_layout()

        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = '../../fig_results/'
        file_path = f'{folder_path}/{"3d"}_{current_time}.pdf'
        # plt.savefig(file_path, dpi=1000, transparent=True, bbox_inches='tight')
        plt.savefig(file_path, dpi=1000, transparent=True)
        plt.show()

    @staticmethod
    def contour_plot(coe, num, lb, ub, template: Templates, lamda, safe_set, xlim=None, ylim=None):
        assert lb.shape[0] == 2
        x = np.linspace(lb[0], ub[0], num)
        y = np.linspace(lb[1], ub[1], num)

        X, Y = np.meshgrid(x, y)
        xy_flat = np.column_stack([X.ravel(), Y.ravel()])

        Z_flat = template.calc_values(xy_flat)
        Z_flat = np.dot(Z_flat, coe)
        Z_flat = 1 - Z_flat - lamda

        Z = Z_flat.reshape(X.shape)

        radius = 1.0
        mask = X ** 2 + Y ** 2 > radius ** 2
        Z = np.ma.array(Z, mask=mask)

        width = 800
        height = 800

        px = 1 / plt.rcParams["figure.dpi"]
        # fig, ax = plt.subplots(figsize=(width * px, height * px), layout="constrained")
        fig, ax = plt.subplots(figsize=(width * px, height * px))
        from matplotlib.colors import LinearSegmentedColormap
        viridis = plt.cm.viridis_r
        colors = viridis(np.linspace(0.3, 1, 256))  # 从30%到100%
        deeper_viridis = LinearSegmentedColormap.from_list('deeper_viridis', colors)

        contour = ax.contour(X, Y, Z, levels=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 1.0],
                             cmap=deeper_viridis, linewidths=1, zorder=100)
        ax.clabel(contour, inline=True, fontsize=26, zorder=100)

        if isinstance(safe_set, Interval):
            ax.add_patch(
                Polygon(
                    safe_set.rectangle(),
                    closed=True,
                    alpha=1,
                    fill=False,
                    linewidth=3,
                    edgecolor="red",
                    facecolor=None,
                    zorder=101
                )
            )
        elif isinstance(safe_set, Ellipsoid):
            theta = np.linspace(0, 2 * np.pi, 1000)
            x = np.cos(theta)
            y = np.sin(theta)
            ax.plot(x, y, linewidth=5, color='red')
        else:
            raise NotImplementedError()

        ax.set_aspect('equal')
        if xlim is not None and ylim is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_xlabel(r'$x$', fontsize=14)
        ax.set_ylabel(r'$y$', fontsize=14)
        ax.grid(True)
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = '../../fig_results/'
        file_path = f'{folder_path}/{"contour_lotka_ours"}_{current_time}.pdf'
        plt.savefig(file_path, dpi=1000, transparent=True, bbox_inches='tight')
        plt.show()

    @staticmethod
    def heatmap_plot(coe, num, lb, ub, template: Templates):
        assert lb.shape[0] == 2
        x = np.linspace(lb[0], ub[0], num)
        y = np.linspace(lb[1], ub[1], num)

        X, Y = np.meshgrid(x, y)
        xy_flat = np.column_stack([X.ravel(), Y.ravel()])

        Z_flat = template.calc_values(xy_flat)
        Z_flat = np.dot(Z_flat, coe)

        Z = Z_flat.reshape(X.shape)

        plt.figure()
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        plt.colorbar(label='z = f(x, y)')

        # plt.title("Contour Plot of z = f(x, y)")
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$y$', fontsize=14)
        # plt.grid(True)
        plt.axis('equal')

        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = '../../fig_results/'
        file_path = f'{folder_path}/{"heatmap"}_{current_time}.pdf'
        plt.savefig(file_path, dpi=1000, transparent=True, bbox_inches='tight')
        plt.show()


