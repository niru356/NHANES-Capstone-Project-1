"""Analyze NHANES adult body measurements for Capstone Project 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DATA_DIR = Path(__file__).resolve().parent
FEMALE_FILE = DATA_DIR / "nhanes_adult_female_bmx_2020.csv"
MALE_FILE = DATA_DIR / "nhanes_adult_male_bmx_2020.csv"
OUTPUT_DIR = DATA_DIR / "capstone_outputs"

MEASUREMENT_NAMES = [
    "Weight (kg)",
    "Height (cm)",
    "Upper arm length (cm)",
    "Upper leg length (cm)",
    "Arm circumference (cm)",
    "Hip circumference (cm)",
    "Waist circumference (cm)",
]
FEMALE_NAMES = MEASUREMENT_NAMES + ["BMI (kg/m^2)"]


def load_measurements(path: Path) -> np.ndarray:
    """Load the comment-prefixed NHANES CSV into a 2D NumPy array."""
    matrix = np.loadtxt(path, delimiter=",", skiprows=19, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(MEASUREMENT_NAMES):
        raise ValueError(
            f"Expected seven measurement columns in {path}; got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite measurements found in {path}")
    return matrix


def describe_weights(label: str, weights: np.ndarray) -> dict[str, float]:
    """Print location, dispersion, and shape statistics and return them."""
    result = {
        "mean": float(np.mean(weights)),
        "median": float(np.median(weights)),
        "variance": float(np.var(weights, ddof=1)),
        "std": float(np.std(weights, ddof=1)),
        "range": float(np.ptp(weights)),
        "iqr": float(stats.iqr(weights)),
        "skewness": float(stats.skew(weights, bias=False)),
    }
    print(f"\n{label} weight statistics (kg)")
    for name, value in result.items():
        print(f"  {name.capitalize():12s}: {value:.3f}")
    return result


def save_weight_histograms(female: np.ndarray, male: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    combined = np.concatenate((female[:, 0], male[:, 0]))
    bins = np.histogram_bin_edges(combined, bins=30)
    axes[0].hist(female[:, 0], bins=bins, color="#c44e8b", edgecolor="white")
    axes[0].set_title("Female weight distribution")
    axes[0].set_ylabel("Count")
    axes[1].hist(male[:, 0], bins=bins, color="#4c72b0", edgecolor="white")
    axes[1].set_title("Male weight distribution")
    axes[1].set_xlabel("Weight (kg)")
    axes[1].set_ylabel("Count")
    fig.suptitle("Adult weight distributions (shared weight scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_weight_boxplot(female: np.ndarray, male: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [female[:, 0], male[:, 0]],
        tick_labels=["Female", "Male"],
        patch_artist=True,
        boxprops={"facecolor": "#b8c9dd"},
        medianprops={"color": "#8c2d04", "linewidth": 2},
    )
    ax.set_title("Weight comparison by sex")
    ax.set_ylabel("Weight (kg)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_boxplot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scatterplot_matrix(data: np.ndarray, names: list[str]) -> None:
    count = len(names)
    fig, axes = plt.subplots(count, count, figsize=(14, 14))
    for row in range(count):
        for column in range(count):
            ax = axes[row, column]
            if row == column:
                ax.hist(data[:, column], bins=24, color="#7895b2", edgecolor="white")
            else:
                ax.scatter(
                    data[:, column],
                    data[:, row],
                    s=5,
                    alpha=0.24,
                    color="#35618f",
                    linewidths=0,
                )
            if row == count - 1:
                ax.set_xlabel(names[column], fontsize=8)
            else:
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel(names[row], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("Female measurements: scatterplot matrix", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "female_scatterplot_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_ratio_boxplot(ratios: list[np.ndarray], labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        ratios,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#c4d9c0"},
        medianprops={"color": "#8c2d04", "linewidth": 2},
    )
    ax.set_title("Waist-to-height and waist-to-hip ratios")
    ax.set_ylabel("Ratio")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ratio_boxplot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_correlations(data: np.ndarray, names: list[str]) -> None:
    pearson = np.corrcoef(data, rowvar=False)
    spearman = stats.spearmanr(data, axis=0).statistic
    print("\nFemale Pearson correlation matrix:")
    print("Rows/columns:", ", ".join(names))
    print(np.array2string(pearson, precision=3, suppress_small=True))
    print("\nFemale Spearman rank-correlation matrix:")
    print("Rows/columns:", ", ".join(names))
    print(np.array2string(spearman, precision=3, suppress_small=True))


def print_weight_comparison(
    female_stats: dict[str, float], male_stats: dict[str, float]
) -> None:
    print("\nWeight comparison")
    mean_difference = male_stats["mean"] - female_stats["mean"]
    median_difference = male_stats["median"] - female_stats["median"]
    print(f"  Male minus female mean:   {mean_difference:+.2f} kg")
    print(f"  Male minus female median: {median_difference:+.2f} kg")
    for sex, result in (("female", female_stats), ("male", male_stats)):
        shape = (
            "right-skewed"
            if result["skewness"] > 0.2
            else "left-skewed"
            if result["skewness"] < -0.2
            else "approximately symmetric"
        )
        print(
            f"  {sex.capitalize()} weights are {shape} "
            f"(skewness {result['skewness']:.2f}); SD "
            f"{result['std']:.2f} kg and IQR {result['iqr']:.2f} kg."
        )
    print(
        "  In each boxplot, the center line is the median, the box is the "
        "interquartile range, and points beyond the whiskers indicate potential "
        "outliers. Compare medians, spreads, skew, and tails rather than treating "
        "the groups as non-overlapping."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # These are the original seven-column NumPy arrays: weight, height, arm/leg
    # lengths, arm circumference, hip circumference, and waist circumference.
    female = load_measurements(FEMALE_FILE)
    male = load_measurements(MALE_FILE)
    print(f"Female array shape (before BMI): {female.shape}")
    print(f"Male array shape:                {male.shape}")
    print("Measurement order:", ", ".join(MEASUREMENT_NAMES))

    save_weight_histograms(female, male)
    save_weight_boxplot(female, male)
    female_stats = describe_weights("Female", female[:, 0])
    male_stats = describe_weights("Male", male[:, 0])
    print_weight_comparison(female_stats, male_stats)

    # Column 8 (index 7) is female BMI; height is converted from cm to metres.
    female_bmi = female[:, 0] / (female[:, 1] / 100) ** 2
    female = np.column_stack((female, female_bmi))
    print(f"\nFemale array shape (BMI added as column 8): {female.shape}")
    print(f"Female BMI mean: {np.mean(female[:, 7]):.2f} kg/m^2")

    means = np.mean(female, axis=0)
    standard_deviations = np.std(female, axis=0, ddof=0)
    if np.any(standard_deviations == 0):
        raise ValueError("Cannot z-score a female measurement with zero variance")
    zfemale = (female - means) / standard_deviations
    print(f"zfemale shape (z-scores for all eight columns): {zfemale.shape}")

    correlation_names = ["Height (cm)", "Weight (kg)", "Waist (cm)", "Hip (cm)", "BMI"]
    correlation_data = female[:, [1, 0, 6, 5, 7]]
    save_scatterplot_matrix(correlation_data, correlation_names)
    print_correlations(correlation_data, correlation_names)
    print(
        "\nCorrelation interpretation: positive values mean that larger values of "
        "one measurement tend to accompany larger values of the other; values near "
        "zero indicate little linear/monotonic association. Pearson measures linear "
        "association and is more sensitive to outliers, while Spearman measures "
        "rank-based monotonic association. Correlation does not establish causation."
    )
    print(
        "  In these female data, weight, waist, hip, and BMI are strongly positively "
        "associated (r about 0.90-0.95; rho about 0.89-0.95). Height is only weakly "
        "related to BMI (r=0.033; rho=0.020), whereas height has a modest positive "
        "association with weight (r=0.345). Pearson and Spearman values are close "
        "for these pairs, suggesting broadly similar linear and rank-order patterns."
    )

    male_waist_height = male[:, 6] / male[:, 1]
    female_waist_height = female[:, 6] / female[:, 1]
    male_waist_hip = male[:, 6] / male[:, 5]
    female_waist_hip = female[:, 6] / female[:, 5]
    ratio_arrays = [
        male_waist_height,
        female_waist_height,
        male_waist_hip,
        female_waist_hip,
    ]
    ratio_labels = [
        "Male waist/height",
        "Female waist/height",
        "Male waist/hip",
        "Female waist/hip",
    ]
    save_ratio_boxplot(ratio_arrays, ratio_labels)
    print("\nRatio summaries (median and IQR)")
    for label, values in zip(ratio_labels, ratio_arrays):
        print(
            f"  {label:20s}: median {np.median(values):.3f}, "
            f"IQR {stats.iqr(values):.3f}"
        )
    print(
        "  A higher waist-to-height value means the waist is larger relative to "
        "stature; a higher waist-to-hip value means the waist is larger relative "
        "to the hip circumference. In this sample, female waist-to-height is "
        "higher at the median (0.610 vs 0.582), while male waist-to-hip is higher "
        "(0.977 vs 0.903). These are descriptive differences; the boxplot also "
        "shows each ratio's spread and outliers."
    )

    low_indices = np.argsort(female[:, 7])[:5]
    high_indices = np.argsort(female[:, 7])[-5:][::-1]
    print("\nFive females with the lowest BMI (index, BMI, standardized measurements)")
    print("Z-score columns:", ", ".join(FEMALE_NAMES))
    for index in low_indices:
        print(f"  row {index:4d} | BMI {female[index, 7]:6.2f} | {zfemale[index]}")
    print("\nFive females with the highest BMI (index, BMI, standardized measurements)")
    print("Z-score columns:", ", ".join(FEMALE_NAMES))
    for index in high_indices:
        print(f"  row {index:4d} | BMI {female[index, 7]:6.2f} | {zfemale[index]}")
    print(
        "  The five lowest BMIs are 14.20-15.42 kg/m^2, with BMI z-scores "
        "approximately -1.89 to -2.05. Their weight, arm circumference, hip, and "
        "waist measurements are also generally well below the sample means; "
        "height varies. The five highest BMIs are 64.20-67.04 kg/m^2, with "
        "BMI z-scores approximately 4.40-4.76. Their weights and circumferences "
        "are markedly above sample means, while their heights are near average. "
        "These are sample-relative patterns, not diagnoses; BMI extremes alone "
        "do not identify health status or body composition."
    )

    print(
        "\nShort discussion of body-size indicators\n"
        "  BMI: Simple, inexpensive, and permits broad population comparisons. "
        "It does not distinguish fat from muscle or describe fat distribution, "
        "and the same BMI can have different implications across individuals.\n"
        "  Waist-to-height ratio: Relates central waist size to stature, is easy "
        "to calculate, and partly adjusts for body size. It depends on consistent "
        "waist measurement technique and does not directly measure body fat or "
        "muscle; interpretation may vary with age and population.\n"
        "  Waist-to-hip ratio: Captures waist size relative to hip circumference "
        "and provides information about body-fat distribution. It requires two "
        "measurements, can change because either measurement changes, and shares "
        "the same measurement and individual-context limitations.\n"
        "  All three are screening/description measures, not stand-alone diagnoses."
    )
    print(f"\nPlots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
