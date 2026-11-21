import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = Path(__file__).resolve().parent.parent / "act_redblue_from_foil"
OUT_DIR = Path(__file__).resolve().parent
RESUME_STEP = 20_000
KAMEL_EVAL_AT_20K = 0.3356
SMOOTH_WINDOW = 5


def read_rows(name):
    with open(RUN_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return [float(r[key]) for r in rows]


def moving_average(values, window):
    half = window // 2
    return [
        sum(values[max(0, i - half) : i + half + 1]) / len(values[max(0, i - half) : i + half + 1])
        for i in range(len(values))
    ]


def mark_resume(ax):
    ax.axvline(RESUME_STEP, color="gray", linestyle="--", linewidth=1)
    ax.text(RESUME_STEP, ax.get_ylim()[1], " resumed at 20K", va="top", fontsize=8, color="gray")


def plot_train(train):
    steps = column(train, "step")
    loss, l1, kld, grad = (column(train, k) for k in ("loss", "l1_loss", "kld_loss", "grdn"))

    fig, (ax_loss, ax_kld, ax_grad) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]}
    )
    ax_loss.plot(steps, loss, color="tab:blue", alpha=0.25, linewidth=1)
    ax_loss.plot(steps, moving_average(loss, SMOOTH_WINDOW), color="tab:blue", label="train loss (L1 + 10·KL)")
    ax_loss.plot(steps, l1, color="tab:orange", alpha=0.25, linewidth=1)
    ax_loss.plot(steps, moving_average(l1, SMOOTH_WINDOW), color="tab:orange", label="train L1")
    ax_loss.set_ylabel("loss (per 100 steps)")
    ax_loss.set_title("act_redblue_from_foil: training loss (lr 2e-5 constant, batch 32)")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    mark_resume(ax_loss)

    ax_kld.plot(steps, kld, color="tab:purple")
    ax_kld.set_ylabel("KL")
    ax_kld.grid(alpha=0.3)
    mark_resume(ax_kld)

    ax_grad.plot(steps, grad, color="tab:green", alpha=0.4, linewidth=1)
    ax_grad.plot(steps, moving_average(grad, SMOOTH_WINDOW), color="tab:green")
    ax_grad.set_ylabel("grad norm")
    ax_grad.set_xlabel("step")
    ax_grad.grid(alpha=0.3)
    mark_resume(ax_grad)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "train_loss.png", dpi=130)


def plot_eval(train, evals):
    l1_by_step = {float(r["step"]): float(r["l1_loss"]) for r in train}
    steps = column(evals, "step")
    eval_loss = column(evals, "eval_loss")
    train_loss = column(evals, "train_loss_at_step")
    train_l1 = [l1_by_step[s] for s in steps]

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax.plot(steps, eval_loss, "o-", color="tab:red", label="eval loss, 4 held-out episodes (L1, zero latent)")
    ax.plot(steps, train_loss, "s-", color="tab:blue", label="train loss at same step (L1 + 10·KL)")
    ax.plot(steps, train_l1, "^-", color="tab:orange", label="train L1 at same step")
    ax.plot([RESUME_STEP], [KAMEL_EVAL_AT_20K], "*", color="black", markersize=12, label="Kamel from scratch, eval @20K")
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("eval vs train loss")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    mark_resume(ax)

    ax_zoom.plot(steps, eval_loss, "o-", color="tab:red", label="from foil (Tomi)")
    ax_zoom.plot([RESUME_STEP], [KAMEL_EVAL_AT_20K], "*", color="black", markersize=12, label="from scratch (Kamel)")
    for s, v in zip(steps, eval_loss):
        ax_zoom.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax_zoom.set_ylim(0.32, 0.345)
    ax_zoom.set_xlabel("step")
    ax_zoom.set_title("eval loss, zoomed (spread 0.009 = 3%)")
    ax_zoom.legend(fontsize=8)
    ax_zoom.grid(alpha=0.3)
    mark_resume(ax_zoom)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eval_vs_train.png", dpi=130)


def main():
    train = read_rows("train_losses.csv")
    evals = read_rows("eval_losses.csv")
    plot_train(train)
    plot_eval(train, evals)


if __name__ == "__main__":
    main()
