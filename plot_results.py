
import ast
from typing import Literal

import colorsys
import matplotlib.pyplot as plt
import polars as pl
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
        show: bool = True,
        loglog: bool = False,
        plot_all: bool = False,
) -> None:
    settings = get_unique_settings(file, ["initial_backward_prob", "adjust_backward_prob", "depth", "width"])
    if initial_backward_prob is not None:
        settings = [(bp, ap, d, w) for bp, ap, d, w in settings if bp == initial_backward_prob]
    if adjust_backward_prob is not None:
        settings = [(bp, ap, d, w) for bp, ap, d, w in settings if ap == adjust_backward_prob]
    if depth is not None:
        settings = [(bp, ap, d, w) for bp, ap, d, w in settings if d == depth]
    if width is not None:
        settings = [(bp, ap, d, w) for bp, ap, d, w in settings if w == width]
    colors = generate_distinct_colors(len(settings)*2)
    col_num = 0

    for (initial_backward_prob_, adjust_backward_prob_, depth_, width_) in settings:
        for direction in ("fw", "bw"):
            color = colors[col_num]
            col_num += 1
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                mask=mask,
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
            
            label = f"{direction}, p_bw=({initial_backward_prob_}), depth={depth_}, width={width_}" + (" (adjusted)" if adjust_backward_prob_ else "")
            if loglog:
                plt.loglog(xs, avg_ys, color=color, label=label, linestyle=linestyle)
            else:
                plt.plot(xs, avg_ys, color=color, label=label, linestyle=linestyle)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.grid()
    plt.title(f"{to_plot} vs {plot_over}")
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    plot_fw_bw(
        file="results/results_scaling_fw_bw.csv",
        to_plot="val_losses",
        plot_over="epoch",
        mask="bidirectional",
        adjust_backward_prob=False,
        initial_backward_prob=0.1,
        depth=8,
        width=None,
        show=True,
        loglog=True,
    )
