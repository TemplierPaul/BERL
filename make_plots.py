import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import pandas as pd
import wandb
from scipy import interpolate
import seaborn as sns
import sys

def get_data(entity, project, jobs=None):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)
    if jobs is not None:
        runs = [r for r in runs if "job" in r.config and int(r.config["job"]) in jobs]

    if min_evals is not None:
        runs = [r for r in runs if r.history()["evaluations"].iloc[-1] >= min_evals]

    data = {}
    for r in runs:
        p = r.config["problem"]
        n = r.config["noise"]
        g = f"{p}_{n*100}"
        a = r.config["algo"]
        if g not in data:
            data[g] = {}
        if a not in data[g]:
            data[g][a] = []
        data[g][a].append(r)
    return data


class GroupedRun:
    def __init__(self, algo, runs):
        self.algo = algo
        self.runs = runs
        self.interpolators = {}
        self.ranges = {}

    # length
    def __len__(self):
        return len(self.runs)

    def interpolate(self, y):
        if y not in self.interpolators:
            self.interpolators[y] = []
            self.ranges[y] = (- np.inf, max_evals)
        for r in self.runs:
            h = r.history()
            i = interpolate.interp1d(h["evaluations"], h[y])
            self.interpolators[y].append(i)
            self.ranges[y] = (max(self.ranges[y][0], h["evaluations"].min()), min(self.ranges[y][1], h["evaluations"].max()))

    def interpolate_cost(self):
        y = "cost"
        if y not in self.interpolators:
            self.interpolators[y] = []
            self.ranges[y] = (- np.inf, max_evals)
        for r in self.runs:
            h = r.history()
            evals = h["evaluations"]
            n_pop = r.config["n_pop"]
            gen = h["generation"]
            cost = evals / (n_pop * gen)
            i = interpolate.interp1d(h["evaluations"], cost)
            self.interpolators[y].append(i)
            self.ranges[y] = (max(self.ranges[y][0], h["evaluations"].min()), min(self.ranges[y][1], h["evaluations"].max()))

    def compute(self, field, x):
        # filter x for range
        x = [x[i] for i in range(len(x)) if self.ranges[field][0] <= x[i] <= self.ranges[field][1]]
        y = []
        for i in self.interpolators[field]:
            y.append(i(x))
        return np.array(x), np.array(y)

    def get_mean_std(self, field, x):
        x, y = self.compute(field, x)
        return np.mean(y, axis=0), np.std(y, axis=0)

    def has_field(self, field):
        # check if runs have field
        for r in self.runs:
            if field not in r.history():
                return False
        return True

    def get_field(self, field):
        # get runs field
        y = []
        for r in self.runs:
            for _ in range(10):
                try:
                    h = r.history()
                    y.append(h[field].iloc[-1])
                    break
                except:
                    pass
        return np.array(y)
            
if __name__ == "__main__":
    sns.set()

    entity = "sureli"
    project = "lucie-robot"

    max_evals = 20000
    min_evals = 10000

    colors = {
        "UCEA":"blue",
        "EA":"red",
        "MultiEA":"green",
    }
    markers =  {
        "UCEA":"^",
        "EA":"o",
        "MultiEA":"s",
    }
    rename = {
        "UCEA":"LUCIE",
        "MultiEA":"Resample",
        # EA becomes "mu plus lambda" in greek letters
        "EA": r"$\mu + \lambda$",
    }

    boxplot_order = ["EA", "MultiEA", "UCEA"]

    # get cli argument
    try:
        env = sys.argv[1].lower()
    except:
        env = "test"
    jobs_dict={
        "test":[10000, 10100, 10101],
        "cartpole":[775104, 775105, 775106, 775108, 775116],
        "acrobot":[775100, 775101, 775102, 775103, 775117, 775437, 775438, 775439, 775117],
        "double_pendulum":[17000, 17001, 17002, 17003, 17004],
        "leading_ones_u":[21000],
        "one_max_u": [21001],
        "leading_ones_g": [21002],
        "one_max_g": [21003],
    }

    noise_type="u"
    if env.endswith("_g"):
        noise_type="g"
    jobs = jobs_dict[env]

    # jobs = None
    # jobs = [10000, 10100, 10101]

    # Cartpole
    # jobs = [775104, 775105, 775106, 775108, 775116]
    # Acrobot
    # jobs = [775100, 775101, 775102, 775103, 775117]
    # double-pendulum
    # jobs = [17000, 17001, 17002, 17003, 17004]

    print("\n" + "-" * 80 + "\n")
    print("GETTING DATA")
    print("-" * 80 + "\n")

    games = get_data(entity, project, jobs)

    print("\n" + "-" * 80 + "\n")
    print("FITNESS PLOTS")
    print("-" * 80 + "\n")

    for g, runs in games.items():
        y = "fitness"
        for a in boxplot_order:
            try:
                r = runs[a]
            except:
                continue
            # Fitness plot
            group = GroupedRun(a, r)
            print(f"{g}_{noise_type} {a} {y} {len(group)}")
            if not group.has_field(y):
                continue
            group.interpolate(y)
            mean, std = group.get_mean_std(y, np.arange(group.ranges[y][0], group.ranges[y][1]))
            # Use LUCIE instead of UCEA as label
            
            label = rename[a] if a in rename else a
            X = np.arange(group.ranges[y][0], group.ranges[y][1]) / 1000
            plt.plot(
                X,
                mean, 
                label=label,
                # color=colors[a],
                marker=markers[a],
                markevery=len(mean) // 10,
                )
            plt.fill_between(X, mean - std, mean + std, alpha=0.2)
        plt.legend()
        # plt.title(f"{g}_{noise_type}")
        # save figure without borders
        plt.savefig(f"paper_figures/{g}_{noise_type}_{y}.png", bbox_inches="tight")
        plt.show()

    # print("\n" + "-" * 80 + "\n")
    # print("COST LUCIE")
    # print("-" * 80 + "\n")


    # for g, runs in games.items():
    #     y = "cost"
    #     a = "UCEA"
    #     try:
    #         r = runs[a]
    #     except:
    #         continue
    #     # Cost plot
    #     group = GroupedRun(a, r)
    #     print(f"{g}_{noise_type} {a} {y} {len(group)}")
    #     group.interpolate_cost()
    #     mean, std = group.get_mean_std(y, np.arange(group.ranges[y][0], group.ranges[y][1]))
    #     # Use LUCIE instead of UCEA as label
        
    #     label = f"{g.split('_')[-1]}%"
    #     X = np.arange(group.ranges[y][0], group.ranges[y][1]) / 1000
    #     plt.plot(
    #         X,
    #         mean, 
    #         label=label,
    #         # color=colors[a],
    #         marker=markers[a],
    #         markevery=len(mean) // 10,
    #         )
    #     plt.fill_between(X, mean - std, mean + std, alpha=0.2)
    # # put legend bottom right
    # plt.legend(loc="lower right")
    # # plt.title(f"{g}_{noise_type}")
    # # save figure without borders
    # try:
    #     plt.savefig(f"paper_figures/cost_LUCIE_{g.split('_')[0]}.png", bbox_inches="tight")
    #     plt.show()
    # except:
    #     print("No plot to save")

    # print("\n" + "-" * 80 + "\n")
    # print("COST PLOTS")
    # print("-" * 80 + "\n")

    # for g, runs in games.items():
    #     y = "cost"
    #     for a in boxplot_order:
    #         try:
    #             r = runs[a]
    #         except:
    #             continue
    #         # Cost plot
    #         group = GroupedRun(a, r)
    #         print(f"{g}_{noise_type} {a} {y} {len(group)}")
    #         group.interpolate_cost()
    #         mean, std = group.get_mean_std(y, np.arange(group.ranges[y][0], group.ranges[y][1]))
    #         # Use LUCIE instead of UCEA as label
            
    #         label = rename[a] if a in rename else a
    #         X = np.arange(group.ranges[y][0], group.ranges[y][1]) / 1000
    #         plt.plot(
    #             X,
    #             mean, 
    #             label=label,
    #             # color=colors[a],
    #             marker=markers[a],
    #             markevery=len(mean) // 10,
    #             )
    #         plt.fill_between(X, mean - std, mean + std, alpha=0.2)
    #     # put legend bottom right
    #     plt.legend(loc="lower right")
    #     # plt.title(f"{g}_{noise_type}")
    #     # save figure without borders
    #     plt.savefig(f"paper_figures/cost_{g}_{noise_type}_{y}.png", bbox_inches="tight")
    #     plt.show()

    print("\n" + "-" * 80 + "\n")
    print("BOXPLOTS")
    print("-" * 80 + "\n")

    sns.set(rc={'figure.figsize':(3, 5)})

    # boxplot_order = ['UCEA', 'MultiEA', 'EA']
    for g, runs in games.items():
        data = []
        # boxplot_order = list(runs.keys())
        print(list(runs.keys()))
        fig, ax = plt.subplots()
        labels = []
        for a in boxplot_order:
            try:
                r = runs[a]
            except:
                continue
            labels.append(rename[a] if a in rename else a)
            # Fitness plot
            group = GroupedRun(a, r)
            print(f"{g}_{noise_type} {a} {len(group)}")
            val = group.get_field(field="final validation fitness")
            # print(val)
            print(len(val), np.mean(val))
            data.append(val)
        sns.boxplot(
                data=data,
                linewidth=2.5,
                width=0.5,
            )
        # set labels 
        ax.set_xticklabels(labels)
        # smaller width
        # ax.tick_params(axis="x", which="major", width=1)

        plt.savefig(f"paper_figures/boxplot_{g}_{noise_type}.png", bbox_inches="tight")  
        plt.show()
        