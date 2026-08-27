import json
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

EVAL_DIR = "./eval_results"

LANG_MAP = {
    "code_c": "C",
    "code_cpp": "C++",
    "code_csharp": "C#",
    "code_go": "Go",
    "code_java": "Java",
    "code_javascript": "JavaScript",
    "code_php": "PHP",
    "code_typescript": "TypeScript",
}

records = []

for subdir, lang_label in LANG_MAP.items():
    path = os.path.join(EVAL_DIR, subdir)

    base_files = sorted(glob.glob(os.path.join(path, "eval_results_base_*.json")))
    ft_files = sorted(glob.glob(os.path.join(path, "eval_results_ft_*.json")))

    if not base_files or not ft_files:
        continue

    # Use the most recent base file
    for label, fpath in [("Base", base_files[-1]), ("Fine-tuned", ft_files[-1])]:
        with open(fpath) as f:
            d = json.load(f)
        results = d["results"]

        humaneval_score = results.get("humaneval", {}).get("pass@1,create_test", None)
        humaneval_stderr = results.get("humaneval", {}).get("pass@1_stderr,create_test", 0.0)
        mbpp_score = results.get("mbpp", {}).get("pass_at_1,none", None)
        mbpp_stderr = results.get("mbpp", {}).get("pass_at_1_stderr,none", 0.0)

        if humaneval_score is not None:
            records.append({
                "Language": lang_label,
                "Model": label,
                "Benchmark": "HumanEval",
                "Pass@1": humaneval_score * 100,
                "Stderr": humaneval_stderr * 100,
            })
        if mbpp_score is not None:
            records.append({
                "Language": lang_label,
                "Model": label,
                "Benchmark": "MBPP",
                "Pass@1": mbpp_score * 100,
                "Stderr": mbpp_stderr * 100,
            })

df = pd.DataFrame(records)
print(df.to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk", font_scale=0.95)
palette = {"Base": "#4C72B0", "Fine-tuned": "#DD8452"}

benchmarks = df["Benchmark"].unique()
fig, axes = plt.subplots(1, len(benchmarks), figsize=(7 * len(benchmarks), 6), sharey=False)
if len(benchmarks) == 1:
    axes = [axes]

for ax, bench in zip(axes, benchmarks):
    sub = df[df["Benchmark"] == bench].copy()
    langs = sub["Language"].unique()

    bar_ax = sns.barplot(
        data=sub,
        x="Language",
        y="Pass@1",
        hue="Model",
        palette=palette,
        order=langs,
        hue_order=["Base", "Fine-tuned"],
        ax=ax,
        width=0.6,
        capsize=0.04,
    )

    # Add error bars manually using Stderr column
    n_langs = len(langs)
    n_models = 2
    bar_width = 0.6 / n_models
    offsets = [-bar_width / 2, bar_width / 2]

    for i, model in enumerate(["Base", "Fine-tuned"]):
        model_sub = sub[sub["Model"] == model].set_index("Language")
        for j, lang in enumerate(langs):
            if lang in model_sub.index:
                val = model_sub.loc[lang, "Pass@1"]
                err = model_sub.loc[lang, "Stderr"]
                x_pos = j + offsets[i]
                ax.errorbar(x_pos, val, yerr=err, fmt="none", color="black",
                            capsize=4, linewidth=1.2, capthick=1.2)

    ax.set_title(bench, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Pass@1 (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Model", frameon=True)

    # Annotate bars with values
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.5:
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

fig.suptitle("Base vs Fine-tuned: Pass@1 by Language", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = "./eval_results/base_vs_finetuned.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot to {out_path}")
