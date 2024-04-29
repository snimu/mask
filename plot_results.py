
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
                f"mask={mask_}, {direction}, p_bw=({initial_backward_prob_}), "
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
) -> tuple[np.ndarray, np.ndarray]:
    xs_fw, _, avg_ys_fw = load_xs_ys_avg_y(
        file,
        mask="forward",
        to_plot=to_plot,
        plot_over=plot_over,
        **settings,
    )
    xs_bw, _, avg_ys_bw = load_xs_ys_avg_y(
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

    fw_bw_ratio = avg_ys_fw / avg_ys_bw

    return chosen_xs, fw_bw_ratio


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


def plot_perf_forward_by_perf_bideirectional_over_num_params(
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
) -> None:
    param_nums = unique_num_params(file)
    widths = unique_widths(file)
    depths = unique_depths(file)
    
    ratios_num_params = []
    for num_params in param_nums:
        _, fw_bw_ratio = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            num_params=num_params, 
        )
        ratios_num_params.append(fw_bw_ratio.tolist())

    ratios_widths = []
    for width in widths:
        _, fw_bw_ratio = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            width=width, 
        )
        ratios_widths.append(fw_bw_ratio.tolist())

    ratios_depths = []
    for depth in depths:
        _, fw_bw_ratio = get_ratio(
            file, 
            to_plot=f"{to_plot}_{direction}", 
            plot_over=plot_over,
            initial_backward_prob=initial_backward_prob,
            adjust_backward_prob=adjust_backward_prob,
            depth=depth, 
        )
        ratios_depths.append(fw_bw_ratio.tolist())

    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2)
    # plt.title(f"{to_plot}_{direction}: ratio forward mask to bidirectional mask by size")

    ax0 = plt.subplot(gs[0, 0])
    ax0.boxplot(ratios_depths, labels=depths)
    ax0.set_xlabel("Depth")
    ax0.set_ylabel(f"{to_plot}-{direction} ratio forward/bidirectional mask")
    ax0.grid()

    ax1 = plt.subplot(gs[0, 1])
    ax1.boxplot(ratios_widths, labels=widths)
    ax1.set_xlabel("Width")
    ax1.grid()

    ax2 = plt.subplot(gs[1, :])
    ax2.boxplot(ratios_num_params, labels=[format_num_params(num) for num in param_nums])
    ax2.set_xlabel("#params")
    ax2.set_ylabel(f"{to_plot}-{direction} ratio forward/bidirectional mask")
    ax2.grid()
    

    plt.tight_layout()
    if show:
        plt.show()
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
    #     depth=None,
    #     width=768,
    #     fw_only=True,
    #     show=True,
    #     loglog=False,
    # )
    # plot_perf_forward_by_perf_bideirectional_over_num_params(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_bakward_prob=False,
    #     moving_avg_window_size=5,
    # )
    # plot_heatmap_depth_width_perf_forward_by_perf_bidirectional(
    #     file=file,
    #     initial_backward_prob=0.05,
    #     adjust_bakward_prob=False,
    #     to_plot="val_losses",
    #     plot_over="epoch",
    # )
    plot_ratio_over_num_params(
        file=file,
        initial_backward_prob=0.1,
        adjust_backward_prob=False,
        to_plot="val_losses",
        plot_over="epoch",
    )
