#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Plot Function
#  Update: 2021-03-10, Yuhang Zhang: Revise Codes


import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from gops.utils.tensorboard_setup import read_tensorboard


def self_plot(
    data,
    fname=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    legend_loc="best",
    color_list=None,
    xlim=None,
    ylim=None,
    xtick=None,
    ytick=None,
    yline=None,
    xline=None,
    ncol=1,
    figsize_scalar=1,
    category="plot",
    use_log_scale=False,
    use_symlog_scale=False,
):
    """
    Plot single figure containing several curves.
    """
    default_cfg = dict()
    default_cfg["fig_size"] = (12, 9)
    default_cfg["dpi"] = 300
    default_cfg["pad"] = 0.5

    default_cfg["tick_size"] = 8
    default_cfg["tick_label_font"] = "Times New Roman"
    default_cfg["legend_font"] = {
        "family": "Times New Roman",
        "size": "8",
        "weight": "normal",
    }
    default_cfg["label_font"] = {
        "family": "Times New Roman",
        "size": "9",
        "weight": "normal",
    }

    default_cfg["img_fmt"] = "png"

    # pre-process
    assert isinstance(data, (dict, list, tuple))

    if isinstance(data, dict):
        data = [data]
    num_data = len(data)

    fig_size = (
        default_cfg["fig_size"] * figsize_scalar,
        default_cfg["fig_size"] * figsize_scalar,
    )
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

    # color list
    if (color_list is None) or len(color_list) < num_data:
        tableau_colors = cycle(mcolors.TABLEAU_COLORS)
        color_list = [next(tableau_colors) for _ in range(num_data)]

    # plot figure
    for (i, d) in enumerate(data):
        if category == "plot":
            plt.plot(d["x"], d["y"], color=color_list[i])
        elif category == "scatter":
            plt.scatter(d["x"], d["y"], color=color_list[0], s=1)
        else:
            raise NotImplemented

    # legend
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]

    if legend is not None:
        plt.legend(legend, loc=legend_loc, ncol=ncol, prop=default_cfg["legend_font"])

    #  label
    plt.xlabel(xlabel, default_cfg["label_font"])
    plt.ylabel(ylabel, default_cfg["label_font"])

    # Set log scale for y-axis if requested
    if use_log_scale:
        ax.set_yscale('log')
    elif use_symlog_scale:
        # Symmetric log scale can handle negative values
        # linthresh determines the range around zero that is linear
        import numpy as np
        if isinstance(data, list):
            all_y = np.concatenate([d["y"] for d in data])
        else:
            all_y = data["y"]
        # Set linthresh to 1% of the maximum absolute value or 1.0, whichever is larger
        linthresh = max(1.0, np.max(np.abs(all_y)) * 0.01)
        ax.set_yscale('symlog', linthresh=linthresh)

    if yline is not None:
        plt.axhline(yline, ls=":", c="grey")
    if xline is not None:
        plt.axvline(xline, ls=":", c="grey")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xtick is not None:
        plt.xticks(xtick)
    if ytick is not None:
        plt.yticks(ytick)
    plt.tight_layout(pad=default_cfg["pad"])

    if fname is None:
        pass
    else:
        plt.savefig(fname)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def plot_all(path):
    data = read_tensorboard(path)
    figure_path = os.path.join(path, "figure")
    os.makedirs(figure_path, exist_ok=True)
    for (key, values) in data.items():
        x_label, y_label = str_edit(key)

        # Handle TAR-related plots
        use_log = False
        use_symlog = False

        if "TAR" in y_label or "TAR" in key:
            import numpy as np
            y_data = values["y"] if isinstance(values, dict) else values[0]["y"]

            # Check data characteristics
            all_positive = np.all(y_data > 0)
            all_negative = np.all(y_data < 0)
            has_mixed_signs = not (all_positive or all_negative)

            if all_positive:
                # All positive: use regular log scale
                use_log = True
                print(f"Using log scale for '{y_label}' (all positive values)")
            elif all_negative:
                # All negative: use linear scale (log scale cannot handle negative values)
                # Keep original negative values for display
                use_log = False
                use_symlog = False
                print(f"Using linear scale for '{y_label}' (all negative values)")
            else:
                # Mixed positive and negative: use symlog
                use_symlog = True
                print(f"Using symlog scale for '{y_label}' (mixed positive/negative values)")

        self_plot(
            values,
            os.path.join(figure_path, x_label + "-" + y_label + ".tiff"),
            xlabel=x_label,
            ylabel=y_label,
            color_list=["orange"],
            use_log_scale=use_log,
            use_symlog_scale=use_symlog,
        )
    plt.show()


def str_edit(str_):
    str_ = str_.replace("\\", "/")
    if "/" in str_:
        str_ = str_.split("/")
        str_ = str_[-1]

    str_total = str_
    x_label = None
    y_label = None
    if "-" in str_:
        str_ = str_.split("-")
        if len(str_) == 2:
            x_label = str_[1]
            y_label = str_[0]
            y_label = re.sub(r"\d\.(\s*)", "", y_label, 1)
    if x_label is None:
        x_label = "Iteration Steps"
        y_label = str_total
    return x_label, y_label
