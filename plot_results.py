
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
import math
import polars as pl
import pandas as pd
import seaborn as sns
import numpy as np


"""Columns: final_val_loss_fw,final_val_loss_bw,mask,model_scale,depth,width,num_params,run_num,seed,adjust_backward_prob,initial_backward_prob,backward_probs,train_losses,train_accs,val_losses_fw,val_accs_fw,val_pplxs_fw,val_losses_bw,val_accs_bw,val_pplxs_bw,tokens_seen_train,epochs_train,steps_train,epoch_by_distinct_tokens_seen_train,tokens_seen_val,epochs_val,steps_val,cumulative_times_val,epoch_by_distinct_tokens_seen_val"""


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        return np.array(ast.literal_eval(series))


def format_num_params(num_params: int) -> str:
    if num_params < 1_000:
        return str(num_params)
    elif num_params < 1_000_000:
        return f"{num_params/1_000:.1f}k"
    elif num_params < 1_000_000_000:
        return f"{num_params/1_000_000:.1f}M"
    else:
        return f"{num_params/1_000_000_000:.1f}B"


def load_xs_ys_avg_y(
        file: str,
        mask: Literal["forward", "backward", "bidirectional"] | None = None,
        initial_backward_prob: float | None = None,
        adjust_backward_prob: bool | None = None,
        model_scale: float | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        to_plot: str = "val_loss_fw",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    filters = (pl.col("mask").is_in(["forward", "backward", "bidirectional"]))  # initial condition -> always true
    if mask is not None:
        filters &= (pl.col("mask") == mask)
    if initial_backward_prob is not None:
        filters &= (pl.col("initial_backward_prob") == initial_backward_prob)
    if adjust_backward_prob is not None:
        filters &= (pl.col("adjust_backward_prob") == adjust_backward_prob)
    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "epoch_unique_token":
        return load_epochs_unique_tokens_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays, to_plot)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])

    if "train" in to_plot:
        xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    elif "val" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 50

    avg_ys = np.mean(ys, axis=0)

    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs_str = "epochs_train" if "train" in to_plot else "epochs_val"
    xs = [series_to_array(df[epochs_str][i]) for i in range(len(df[epochs_str]))]
    return interpolate_linearly(xs, arrays)


def load_epochs_unique_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs_str = "epoch_by_distinct_tokens_seen_train" if "train" in to_plot else "epoch_by_distinct_tokens_seen_val"
    xs = [series_to_array(df[epochs_str][i]) for i in range(len(df[epochs_str]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_str = "tokens_seen_train" if "train" in to_plot else "tokens_seen_val"
    xs = [series_to_array(df[tokens_str][i]) for i in range(len(df[tokens_str]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert "val" in to_plot, "Only validation data has time data"
    time_str = "cumulative_time"
    xs = [series_to_array(df[time_str][i]) for i in range(len(df[time_str]))]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
        .select(*[pl.col(target) for target in targets])
        .collect()
        .unique()
    )
    # Sort combinations alphabetically by content, target by target (for consistency in plotting)
    for target in targets:
        combinations = combinations.sort(target)
    # Create a list of settings
    for features in zip(
            *[combinations[target] for target in targets]
    ):
        settings.append(tuple(features))

    return settings


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def plot_fw_bw(
        file: str,
        to_plot: Literal["val_losses", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_losses",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        mask: Literal["forward", "backward", "bidirectional"] | None = None,
        initial_backward_prob: float | None = None,
        adjust_backward_prob: bool | None = None,
        depth: int | None = 8,
        width: int | None = 384,
        fw_only: bool = False,
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
) -> None:
    settings = get_unique_settings(file, ["mask", "initial_backward_prob", "adjust_backward_prob", "depth", "width"])

    if mask is not None:
        settings = [(ma, bp, ap, d, w) for ma, bp, ap, d, w in settings if ma == mask]
    if initial_backward_prob is not None:
        settings = [(ma, bp, ap, d, w) for ma, bp, ap, d, w in settings if bp == initial_backward_prob or ma == "forward"]
    if adjust_backward_prob is not None:
        settings = [(ma, bp, ap, d, w) for ma, bp, ap, d, w in settings if ap == adjust_backward_prob]
    if depth is not None:
        settings = [(ma, bp, ap, d, w) for ma, bp, ap, d, w in settings if d == depth]
    if width is not None:
        settings = [(ma, bp, ap, d, w) for ma, bp, ap, d, w in settings if w == width]

    colors = generate_distinct_colors(len(settings)*2)
    col_num = 0

    for (mask_, initial_backward_prob_, adjust_backward_prob_, depth_, width_) in settings:
        for direction in ("fw",) if fw_only else ("fw", "bw"):
            color = colors[col_num]
            col_num += 1
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                mask=mask_,
                initial_backward_prob=initial_backward_prob_,
                depth=depth_,
                width=width_,
                to_plot=to_plot+f"_{direction}",
                plot_over=plot_over,
                adjust_backward_prob=adjust_backward_prob_,
            )
            linestyle = "-" if direction == "fw" else "--"
            if plot_all:
                for y in ys:
                    if loglog:
                        plt.loglog(xs, y, color=color, alpha=0.2, linestyle=linestyle)
                    else:
                        plt.plot(xs, y, color=color, alpha=0.2, linestyle=linestyle)

            num_params = pl.scan_csv(file).filter(
                (pl.col("mask") == mask_)
                & (pl.col("initial_backward_prob") == (0.0 if mask_ == "forward" else initial_backward_prob_))
                & (pl.col("adjust_backward_prob") == (False if mask_ == "forward" else adjust_backward_prob_))
                & (pl.col("depth") == depth_)
                & (pl.col("width") == width_)
            ).collect()["num_params"][0]
            
            label = (
                f"{direction}-perf, p_bw=({initial_backward_prob_}), "
                f"depth={depth_}, width={width_}, #params={format_num_params(num_params)}"
                f"{' (adjusted)' if adjust_backward_prob_ else ''}"
            )
            if loglog:
                plt.loglog(xs, avg_ys, color=color if plot_all else None, label=label, linestyle=linestyle)
            else:
                plt.plot(xs, avg_ys, color=color if plot_all else None, label=label, linestyle=linestyle)


    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over}")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        variable_settings = ""
        fixed_settings = ""
        if mask is None:
            variable_settings += "_mask"
        else:
            fixed_settings += f"_{mask=}"
        if initial_backward_prob is None:
            variable_settings += "_initial_backward_prob"
        else:
            fixed_settings += f"_{initial_backward_prob=}"
        if depth is None:
            variable_settings += "_depth"
        else:
            fixed_settings += f"_{depth=}"
        if width is None:
            variable_settings += "_width"
        else:
            fixed_settings += f"_{width=}"

        fixed_settings += f"{'_fw_only' if fw_only else ''}{'_loglog' if loglog else ''}{'_allcurves' if plot_all else ''}"

        savefile = "fw_bw" + variable_settings + fixed_settings + ".png"
        plt.savefig(f"results/images/{savefile}", dpi=300)
    close_plt()


def plot_heatmap_depth_width_perf_forward_by_perf_bidirectional(
        file: str,
        initial_backward_prob: float,
        adjust_bakward_prob: bool,
        to_plot: Literal["val_losses", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_losses",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        show: bool = True,
): 
    df = pd.read_csv(file)
    df_fw = df[df["mask"] == "forward"]
    df_bd = df[
        (df["mask"] == "bidirectional") 
        & (df["initial_backward_prob"] == initial_backward_prob) 
        & (df["adjust_backward_prob"] == adjust_bakward_prob)
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    depths_fw = df_fw["depth"].unique()
    widths_fw = df_fw["width"].unique()

    depths_bd = df_bd["depth"].unique()
    widths_bd = df_bd["width"].unique()

    data_fw = {"depth": [], "width": [], "ratio": []}
    for depth in depths_fw:
        for width in widths_fw:
            xs_fw, _, avg_ys_fw = load_xs_ys_avg_y(
                file,
                mask="forward",
                depth=depth,
                width=width,
                to_plot=f"{to_plot}_fw",
                plot_over=plot_over,
            )
            xs_bd, _, avg_ys_bd = load_xs_ys_avg_y(
                file,
                mask="bidirectional",
                initial_backward_prob=initial_backward_prob,
                adjust_backward_prob=adjust_bakward_prob,
                depth=depth,
                width=width,
                to_plot=f"{to_plot}_fw",
                plot_over=plot_over,
            )
            chosen_xs = min([xs_fw, xs_bd], key=lambda xs: max(xs))
            avg_ys_fw = np.interp(chosen_xs, xs_fw, avg_ys_fw)
            avg_ys_bd = np.interp(chosen_xs, xs_bd, avg_ys_bd)

            fw_bd_ratio = avg_ys_fw.min() / avg_ys_bd.min()
            data_fw["depth"].append(depth)
            data_fw["width"].append(width)
            data_fw["ratio"].append(fw_bd_ratio.item())

    data_bw = {"depth": [], "width": [], "ratio": []}
    for depth in depths_bd:
        for width in widths_bd:
            xs_fw, _, avg_ys_fw = load_xs_ys_avg_y(
                file,
                mask="forward",
                depth=depth,
                width=width,
                to_plot=f"{to_plot}_bw",
                plot_over=plot_over,
            )
            xs_bd, _, avg_ys_bd = load_xs_ys_avg_y(
                file,
                mask="bidirectional",
                initial_backward_prob=initial_backward_prob,
                adjust_backward_prob=adjust_bakward_prob,
                depth=depth,
                width=width,
                to_plot=f"{to_plot}_bw",
                plot_over=plot_over,
            )
            chosen_xs = min([xs_fw, xs_bd], key=lambda xs: max(xs))
            avg_ys_fw = np.interp(chosen_xs, xs_fw, avg_ys_fw)
            avg_ys_bd = np.interp(chosen_xs, xs_bd, avg_ys_bd)

            fw_bd_ratio = avg_ys_fw.min() / avg_ys_bd.min()
            data_bw["depth"].append(depth)
            data_bw["width"].append(width)
            data_bw["ratio"].append(fw_bd_ratio.item())

    pivot_table_fw = pd.pivot_table(pd.DataFrame(data_fw), values="ratio", index="depth", columns="width")
    pivot_table_bd = pd.pivot_table(pd.DataFrame(data_bw), values="ratio", index="depth", columns="width")

    sns.heatmap(pivot_table_fw, ax=axs[0], cmap="coolwarm", annot=True, fmt=".4f", cbar=True, annot_kws={"color": "white"})
    axs[0].set_title(f"{to_plot}_fw ratio forward/bidirectional mask")
    axs[0].set_xlabel("Width")
    axs[0].set_ylabel("Depth")

    sns.heatmap(pivot_table_bd, ax=axs[1], cmap="coolwarm", annot=True, fmt=".4f", cbar=True, annot_kws={"color": "white"})
    axs[1].set_title(f"{to_plot}_bw ratio forward/bidirectional mask")
    axs[1].set_xlabel("Width")
    axs[1].set_ylabel("Depth")

    plt.tight_layout()
    if show:
        plt.show()
    close_plt()


def get_ratio(
        file: str, 
        initial_backward_prob: float | None,
        adjust_backward_prob: bool | None,
        to_plot: str,
        plot_over: str,
        **settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_fw, ys_fw, avg_ys_fw = load_xs_ys_avg_y(
        file,
        mask="forward",
        to_plot=to_plot,
        plot_over=plot_over,
        **settings,
    )
    xs_bw, ys_bw, avg_ys_bw = load_xs_ys_avg_y(
        file,
        mask="bidirectional",
        initial_backward_prob=initial_backward_prob,
        adjust_backward_prob=adjust_backward_prob,
        to_plot=to_plot,
        plot_over=plot_over,
        **settings,
    )
    # Linearly interpolate such that the x-values are guaranteed to align
    # choose the x-values that go the lowest epoch/step/... (though they should be the same)
    chosen_xs = min([xs_fw, xs_bw], key=lambda xs: max(xs))  
    avg_ys_fw = np.interp(chosen_xs, xs_fw, avg_ys_fw)
    avg_ys_bw = np.interp(chosen_xs, xs_bw, avg_ys_bw)
    ys_fw_ = np.empty((len(ys_fw), len(chosen_xs)))
    ys_bw_ = np.empty((len(ys_bw), len(chosen_xs)))
    for i in range(len(ys_fw)):
        ys_fw_[i] = np.interp(chosen_xs, xs_fw, ys_fw[i])
        ys_bw_[i] = np.interp(chosen_xs, xs_bw, ys_bw[i])

    avg_fw_bw_ratio = avg_ys_fw / avg_ys_bw
    fw_bw_ratios = ys_fw_ / ys_bw_

    return chosen_xs, fw_bw_ratios, avg_fw_bw_ratio


def unique_num_params(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("num_params")
        .collect()
        ["num_params"]
        .unique()
        .to_numpy()
    )


def unique_widths(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("width")
        .collect()
        ["width"]
        .unique()
        .to_numpy()
    )


def unique_depths(file: str) -> list[int]:
    return (
        pl.scan_csv(file)
        .select("depth")
        .collect()
        ["depth"]
        .unique()
        .to_numpy()
    )


def plot_perf_forward_by_perf_bidirectional_over_num_params(
        file: str,
        initial_backward_prob: float,
        adjust_backward_prob: bool,
        to_plot: Literal["val_losses", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_losses",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        direction: Literal["fw", "bw"] = "fw",
        show: bool = True,
        moving_avg_window_size: int = 1,
) -> None:
    settings = get_unique_settings(file, ["num_params"])

    for (num_params,) in settings:
        xs_fw, ys_fw, avg_ys_fw = load_xs_ys_avg_y(
            file,
            mask="forward",
            num_params=num_params,
            to_plot=f"{to_plot}_{direction}",
            plot_over=plot_over,
        )
        xs_bw, ys_bw, avg_ys_bw = load_xs_ys_avg_y(
            file,
            mask="bidirectional",
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            num_params=num_params,
            to_plot=f"{to_plot}_{direction}",
            plot_over=plot_over,
        )
        # Linearly interpolate such that the x-values are guaranteed to align
        # choose the x-values that go the lowest epoch/step/... (though they should be the same)
        chosen_xs = min([xs_fw, xs_bw], key=lambda xs: max(xs))  
        avg_ys_fw = np.interp(chosen_xs, xs_fw, avg_ys_fw)
        avg_ys_bw = np.interp(chosen_xs, xs_bw, avg_ys_bw)

        fw_bw_ratio = avg_ys_fw / avg_ys_bw

        moving_averages = np.convolve(fw_bw_ratio, np.ones(moving_avg_window_size)/moving_avg_window_size, mode='valid')
        # Pad the moving averages to aling with the x-values
        moving_averages = np.concatenate((fw_bw_ratio[:moving_avg_window_size-1], moving_averages))

        plt.plot(chosen_xs, moving_averages, label=f"{format_num_params(num_params)}")

    plt.title(f"{to_plot}_{direction}: ratio forward- to bidirectional mask by #params")
    plt.xlabel(plot_over)
    plt.ylabel(f"{to_plot}-{direction} ratio forward/bidirectional mask")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    close_plt()


def plot_ratio_over_num_params(
        file: str,
        initial_backward_prob: float,
        adjust_backward_prob: bool,
        to_plot: Literal["val_losses", "train_losses", "val_accs", "train_accs", "val_pplxs"] = "val_losses",
        plot_over: Literal["step", "epoch", "epoch_unique_token", "token", "time_sec"] = "epoch",
        direction: Literal["fw", "bw"] = "fw",
        show: bool = True,
        from_x_val: int | float = 0,
        to_x_val: int | float | None = None,
) -> None:
    param_nums = unique_num_params(file)
    widths = unique_widths(file)
    depths = unique_depths(file)

    ratios_num_params = []
    labels_num_params = []
    for num_params in param_nums:
        x_vals, fw_bw_ratios, _ = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            num_params=num_params, 
        )
        from_idx = np.where(x_vals >= from_x_val)[0][0]
        to_idx = None if to_x_val is None else np.where(x_vals <= to_x_val)[0][-1]
        ratios = fw_bw_ratios[:, from_idx:to_idx].flatten().tolist()
        ratios_num_params.extend(ratios)
        labels_num_params.extend([format_num_params(num_params)] * len(ratios))

    ratios_widths = []
    labels_widths = []
    for width in widths:
        _, fw_bw_ratios, _ = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            width=width, 
        )
        from_idx = np.where(x_vals >= from_x_val)[0][0]
        to_idx = None if to_x_val is None else np.where(x_vals <= to_x_val)[0][-1]
        ratios = fw_bw_ratios[:, from_idx:to_idx].flatten().tolist()
        ratios_widths.extend(ratios)
        labels_widths.extend([width] * len(ratios))

    ratios_depths = []
    labels_depths = []
    for depth in depths:
        _, fw_bw_ratios, _ = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            depth=depth, 
        )
        from_idx = np.where(x_vals >= from_x_val)[0][0]
        to_idx = None if to_x_val is None else np.where(x_vals <= to_x_val)[0][-1]
        ratios = fw_bw_ratios[:, from_idx:to_idx].flatten().tolist()
        ratios_depths.extend(ratios)
        labels_depths.extend([depth] * len(ratios))

    plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2)

    ax0 = plt.subplot(gs[0, 0])
    df_depths = pd.DataFrame({'Depth': labels_depths, f'{to_plot}_{direction} Ratio': ratios_depths})
    sns.violinplot(x='Depth', y=f'{to_plot}_{direction} Ratio', data=df_depths, ax=ax0, inner=None)
    sns.pointplot(x='Depth', y=f'{to_plot}_{direction} Ratio', data=df_depths, ax=ax0, estimator='median', color='k')
    ax0.set_xlabel("Depth")
    ax0.set_ylabel(f"{to_plot}_{direction}: p_bw=0.0 / p_bw={initial_backward_prob}")
    ax0.grid()

    ax1 = plt.subplot(gs[0, 1])
    df_widths = pd.DataFrame({'Width': labels_widths, f'{to_plot}_{direction} Ratio': ratios_widths})
    sns.violinplot(x='Width', y=f'{to_plot}_{direction} Ratio', data=df_widths, ax=ax1, inner=None)
    sns.pointplot(x='Width', y=f'{to_plot}_{direction} Ratio', data=df_widths, ax=ax1, estimator='median', color='k')
    ax1.set_xlabel("Width")
    ax1.grid()

    ax2 = plt.subplot(gs[1, :])
    df_num_params = pd.DataFrame({'#params': labels_num_params, f'{to_plot}_{direction} Ratio': ratios_num_params})
    sns.violinplot(x='#params', y=f'{to_plot}_{direction} Ratio', data=df_num_params, ax=ax2, inner=None)
    sns.pointplot(x='#params', y=f'{to_plot}_{direction} Ratio', data=df_num_params, ax=ax2, estimator='median', color='k')
    ax2.set_xlabel("#params")
    ax2.set_ylabel(f"{to_plot}_{direction}: p_bw=0.0 / p_bw={initial_backward_prob}")
    ax2.grid()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f"results/images/violinplot_ratio_by_num_params_{to_plot}_{plot_over}_start_{from_x_val}_stop_{to_x_val}.png", dpi=300)
    close_plt()



def get_contrasting_text_color(background_color):
    """Determine whether white or black text would have more contrast against the background color."""
    r, g, b, _ = background_color
    brightness = (r * 333 + g * 333 + (math.sqrt(r*g) * 333)) / 1000
    return 'white' if brightness < 0.75 else 'black'  # Adjust the threshold for more white text

def plot_fw_vs_bw_perf_with_bidirectional_mask_over_number_of_layer_remaining(
        file: str,
        initial_backward_prob: float,
        adjust_backward_prob: bool = False,
        to_plot: Literal[
            "cut_accs", 
            "cut_losses", 
            "cut_pplxs", 
        ] = "cut_losses",
        calculation_order: Literal["mean_then_ratio", "ratio_then_mean"] = "ratio_then_mean",
        show: bool = True,
) -> None:
    param_nums = unique_num_params(file)

    fig, axs = plt.subplots(len(param_nums), 1, figsize=(12, 8), sharex=True)
    bar_width = 0.95  # Set the bar width to 0.95
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    metric_for_final_layer = "val_" + to_plot.split("_")[1]

    all_ratios = []
    all_num_layers_remaining = []
    for i in range(len(param_nums)):
        df_params = (
            pl.scan_csv(file)
            .filter(
                (pl.col("num_params") == param_nums[i])
                & (pl.col("mask") == "bidirectional")
                & (pl.col("initial_backward_prob") == initial_backward_prob)
                & (pl.col("adjust_backward_prob") == adjust_backward_prob)
            )
            .select(
                pl.col("n_layers_removed"), 
                pl.col(to_plot + "_fw"), 
                pl.col(to_plot + "_bw"),
                pl.col(metric_for_final_layer + "_fw"),
                pl.col(metric_for_final_layer + "_bw"),
            )
            .collect()
        )
        n_layers_removed = []
        fw = []
        bw = []
        for row in df_params.iter_rows():
            # Always add the metric without removing any layers (which I stupidly didn't record under the same key)
            n_layers_removed.append([0] + ast.literal_eval(row[0]))
            fw.append([ast.literal_eval(row[3])[-1]] + ast.literal_eval(row[1]))
            bw.append([ast.literal_eval(row[4])[-1]] + ast.literal_eval(row[2]))

        n_layers_removed = np.array(n_layers_removed).mean(axis=0).astype(np.int64)  # throw error in mean if shapes don't fit
        num_layers_remaining = max(n_layers_removed) - n_layers_removed + 1  # Calculate number of layers remaining
        if calculation_order == "mean_then_ratio":
            ratios = np.array(fw).mean(axis=0) / np.array(bw).mean(axis=0)
        elif calculation_order == "ratio_then_mean":
            ratios = np.mean(np.array(fw) / np.array(bw), axis=0)
        else:
            raise ValueError("calculation_order must be 'mean then ratio' or 'ratio then mean'")

        all_ratios.append(ratios)
        all_num_layers_remaining.append(num_layers_remaining)

    flat_ratios = np.concatenate(all_ratios)
    cmap_norm = mcolors.Normalize(vmin=flat_ratios.min(), vmax=flat_ratios.max())

    for i, (ratios, num_layers_remaining) in enumerate(zip(all_ratios, all_num_layers_remaining)):
        colors = [cmap(cmap_norm(ratio)) for ratio in ratios]

        for j in range(len(colors)):
            axs[i].bar(num_layers_remaining[j], 1, width=bar_width, color=colors[j], align='center')
            text_color = get_contrasting_text_color(colors[j])
            axs[i].text(num_layers_remaining[j], 1, f'{ratios[j]:.1f}', ha='center', va='top', color=text_color, fontsize=10)

        # Label the axis with the number of parameters
        axs[i].text(1.02, 0.5, format_num_params(param_nums[i]), va='center', ha='left', transform=axs[i].transAxes)
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Adjust x-axis limits dynamically
        axs[i].set_xlim(0.5, max(num_layers_remaining) + 0.5)

    plt.subplots_adjust(hspace=0.5, left=0.03, right=0.9, bottom=0.25, top=0.85)  # Adjust the layout

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    sm.set_array([])

    # Add colorbar below the entire plot
    cbar_ax = fig.add_axes([0.03, 0.1, 0.85, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'fw/bw {metric_for_final_layer} ratio')

    # Label the x-axis
    fig.text(0.5, 0.2, 'num_layers_remaining', ha='center', va='center')

    # Add main title
    fig.suptitle(f'fw/bw {metric_for_final_layer} ratio for p_bw={initial_backward_prob} ({calculation_order})', fontsize=16)

    # Add label for parameter numbers above them
    fig.text(0.94, 0.88, '#params', ha='center', va='center', fontsize=12, rotation=0)

    if show:
        plt.show()
    else:
        savefile = f"fw_vs_bw_perf_with_bidirectional_mask_over_number_of_layer_remaining_{to_plot}_{calculation_order}.png"
        plt.savefig(f"results/images/{savefile}", dpi=300)
    close_plt()


def plot_fw_perf_with_fw_vs_bidirectional_mask_over_number_of_layer_remaining(
        file: str,
        initial_backward_prob: float,
        adjust_backward_prob: bool = False,
        to_plot: Literal[
            "cut_accs", 
            "cut_losses", 
            "cut_pplxs", 
        ] = "cut_losses",
        calculation_order: Literal["mean_then_ratio", "ratio_then_mean"] = "ratio_then_mean",
        show: bool = True,
) -> None:
    param_nums = unique_num_params(file)

    fig, axs = plt.subplots(len(param_nums), 1, figsize=(12, 8), sharex=True)
    bar_width = 0.95  # Set the bar width to 0.95
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap

    metric_for_final_layer = "val_" + to_plot.split("_")[1]

    all_ratios = []
    all_num_layers_remaining = []
    for i in range(len(param_nums)):
        df_bidir = (
            pl.scan_csv(file)
            .filter(
                (pl.col("num_params") == param_nums[i])
                & (pl.col("mask") == "bidirectional")
                & (pl.col("initial_backward_prob") == initial_backward_prob)
                & (pl.col("adjust_backward_prob") == adjust_backward_prob)
            )
            .select(
                pl.col("n_layers_removed"), 
                pl.col(to_plot + "_fw"), 
                pl.col(metric_for_final_layer + "_fw"),
            )
            .collect()
        )
        n_layers_removed = []
        perf_bidir = []
        for row in df_bidir.iter_rows():
            # Always add the metric without removing any layers (which I stupidly didn't record under the same key)
            n_layers_removed.append([0] + ast.literal_eval(row[0]))
            perf_bidir.append([ast.literal_eval(row[2])[-1]] + ast.literal_eval(row[1]))

        n_layers_removed = np.array(n_layers_removed).mean(axis=0)
        num_layers_remaining = max(n_layers_removed) - n_layers_removed + 1  # Calculate number of layers remaining

        df_fw = (
            pl.scan_csv(file)
            .filter(
                (pl.col("num_params") == param_nums[i])
                & (pl.col("mask") == "forward")
            )
            .select(
                pl.col("n_layers_removed"), 
                pl.col(to_plot + "_fw"), 
                pl.col(metric_for_final_layer + "_fw"),
            )
            .collect()
        )
        perf_fw = []
        for row in df_fw.iter_rows():
            perf_fw.append([ast.literal_eval(row[2])[-1]] + ast.literal_eval(row[1]))
        
        if calculation_order == "mean_then_ratio":
            ratios = np.array(perf_fw).mean(axis=0) / np.array(perf_bidir).mean(axis=0)
        elif calculation_order == "ratio_then_mean":
            ratios = np.mean(np.array(perf_fw) / np.array(perf_bidir), axis=0)
        else:
            raise ValueError("calculation_order must be 'mean then ratio' or 'ratio then mean'")
        
        all_ratios.append(ratios)
        all_num_layers_remaining.append(num_layers_remaining)

    cmap_norm = mcolors.Normalize(vmin=min([ratios.min() for ratios in all_ratios]), vmax=max([ratios.max() for ratios in all_ratios]))  # Normalize for the color map

    for i, (ratios, num_layers_remaining) in enumerate(zip(all_ratios, all_num_layers_remaining)):
        colors = [cmap(cmap_norm(ratio)) for ratio in ratios]

        for j in range(len(colors)):
            axs[i].bar(num_layers_remaining[j], 1, width=bar_width, color=colors[j], align='center')
            text_color = get_contrasting_text_color(colors[j])
            axs[i].text(num_layers_remaining[j], 1, f'{ratios[j]:.1f}', ha='center', va='top', color=text_color, fontsize=10)

        axs[i].text(1.02, 0.5, format_num_params(param_nums[i]), va='center', ha='left', transform=axs[i].transAxes)
        axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        # Adjust x-axis limits dynamically
        axs[i].set_xlim(0.5, max(num_layers_remaining) + 0.5)

    plt.subplots_adjust(hspace=0.5, left=0.03, right=0.9, bottom=0.25, top=0.85)  # Adjust the layout

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    sm.set_array([])

    # Add colorbar below the entire plot
    cbar_ax = fig.add_axes([0.03, 0.1, 0.85, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'fw {metric_for_final_layer}: ratio p_bw=0.0 / p_bw={initial_backward_prob}')

    # Label the x-axis
    fig.text(0.5, 0.2, 'num_layers_remaining', ha='center', va='center')

    # Add main title
    fig.suptitle(f'fw {metric_for_final_layer}: p_bw=0 / p_bw={initial_backward_prob} ({calculation_order})', fontsize=16)

    # Add label for parameter numbers above them
    fig.text(0.94, 0.88, '#params', ha='center', va='center', fontsize=12, rotation=0)

    if show:
        plt.show()
    else:
        savefile = f"fw_{to_plot}_with_fw_vs_bidirectional_mask_over_number_of_layer_remaining_{calculation_order}.png"
        plt.savefig(f"results/images/{savefile}", dpi=300)
    close_plt()


if __name__ == "__main__":
    # file = "results/results_scaling_fw_bw.csv"
    file = "results/results_scaling_with_special_tokens_fw_bw.csv"
    # plot_fw_bw(
    #     file=file,
    #     to_plot="val_losses",
    #     plot_over="epoch",
    #     mask=None,
    #     adjust_backward_prob=False,
    #     initial_backward_prob=0.05,
    #     depth=32,
    #     width=None,
    #     fw_only=True,
    #     show=True,
    #     loglog=False,
    #     plot_all=False,
    # )
    # plot_perf_forward_by_perf_bideirectional_over_num_params(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_backward_prob=False,
    #     moving_avg_window_size=5,
    # )
    # plot_heatmap_depth_width_perf_forward_by_perf_bidirectional(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_bakward_prob=False,
    #     to_plot="val_losses",
    #     plot_over="epoch",
    # )
    # plot_ratio_over_num_params(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_backward_prob=False,
    #     to_plot="val_losses",
    #     direction="fw",
    #     plot_over="epoch",
    #     show=False,
    #     from_x_val=0,
    #     to_x_val=1,
    # )
    # plot_fw_vs_bw_perf_with_bidirectional_mask_over_number_of_layer_remaining(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_backward_prob=False,
    #     to_plot="cut_losses",
    #     calculation_order="ratio_then_mean",
    #     show=False,
    # )
    plot_fw_perf_with_fw_vs_bidirectional_mask_over_number_of_layer_remaining(
        file=file,
        initial_backward_prob=0.05,
        adjust_backward_prob=False,
        to_plot="cut_pplxs",
        calculation_order="mean_then_ratio",
        show=False,
    )
