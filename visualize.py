#!/usr/bin/env python3
"""
AHC062 SA Optimization Visualizer
========================================
Usage:
  python3 visualize.py [test_id] [output]

  test_id : 4-digit case number (default: 0000)
  output  : .gif or .mp4 path; omit to display interactively

Examples:
  python3 visualize.py 0000              # show window
  python3 visualize.py 0000 out.gif      # save GIF
  python3 visualize.py 0000 out.mp4      # save video (requires ffmpeg)

Legend:
  Path panels  : heatmap of visit order (dark=early, bright=late)
                 dashed white line = current path  (sparse)
                 solid cyan line   = best path     (sparse)
  Score graph  : solid red   = best score so far
                 dashed blue = current score
  Temp graph   : green curve = SA temperature (log scale)
  Dotted lines : orange / red = SA restart points (60% / 80%)
"""

import sys, os, subprocess, struct, tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ── helpers ──────────────────────────────────────────────────────────────────

def read_input(test_id: str):
    path = f"tools/tools/in/{test_id}.txt"
    with open(path) as f:
        tokens = f.read().split()
    n = int(tokens[0])
    a = np.array(tokens[1:], dtype=np.int32).reshape(n, n)
    return n, a, path


def run_solver(in_path: str, vis_path: str):
    env = os.environ.copy()
    env["AHC_VIS_FILE"] = vis_path
    subprocess.run(
        ["./target/release/ahc062"],
        stdin=open(in_path, "rb"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,   # suppress solver log
        env=env,
        check=True,
    )


def read_frames(vis_path: str, n: int):
    n2 = n * n
    hdr  = 8 * 4               # elapsed(f64) temp(f64) cur(i64) best(i64)
    body = n2 * 2 * 2          # current path + best path (each: n2 * 2 bytes)
    fsz  = hdr + body

    with open(vis_path, "rb") as f:
        data = f.read()

    frames = []
    off = 0
    while off + fsz <= len(data):
        elapsed,    = struct.unpack_from("<d", data, off); off += 8
        temp,       = struct.unpack_from("<d", data, off); off += 8
        cur_score,  = struct.unpack_from("<q", data, off); off += 8
        best_score, = struct.unpack_from("<q", data, off); off += 8
        cur_p  = np.frombuffer(data, dtype=np.uint8,
                               count=n2*2, offset=off).reshape(n2, 2).copy(); off += n2*2
        best_p = np.frombuffer(data, dtype=np.uint8,
                               count=n2*2, offset=off).reshape(n2, 2).copy(); off += n2*2
        frames.append((elapsed, temp, cur_score, best_score, cur_p, best_p))
    return frames


def visit_order_img(path: np.ndarray, n: int) -> np.ndarray:
    """Return n×n float32 image; value = visit-order / (n2-1)."""
    n2 = n * n
    img = np.empty(n2, dtype=np.float32)
    img[path[:, 0].astype(np.int32) * n + path[:, 1].astype(np.int32)] = (
        np.arange(n2, dtype=np.float32) / (n2 - 1)
    )
    return img.reshape(n, n)


# ── main animation builder ────────────────────────────────────────────────────

def make_animation(frames, n):
    n2 = n * n

    # time-series arrays
    times       = np.array([f[0] for f in frames])
    temps       = np.array([f[1] for f in frames])
    cur_scores  = np.array([f[2] // n2 for f in frames], dtype=np.int64)
    best_scores = np.array([f[3] // n2 for f in frames], dtype=np.int64)

    # SA restart milestones (estimated from relative elapsed times)
    t0, t1 = times[0], times[-1]
    r1_ms = t0 + (t1 - t0) * 0.60
    r2_ms = t0 + (t1 - t0) * 0.80

    # sparse path sample for line overlay
    stride = max(1, n2 // 600)
    idxs   = np.arange(0, n2, stride)

    # ── figure layout ──
    fig = plt.figure(figsize=(17, 8), facecolor="#1a1a2e")
    gs  = GridSpec(2, 3, figure=fig,
                   width_ratios=[2.5, 2.5, 2],
                   hspace=0.38, wspace=0.30,
                   left=0.03, right=0.97, top=0.91, bottom=0.09)

    ax_cur   = fig.add_subplot(gs[:, 0])
    ax_best  = fig.add_subplot(gs[:, 1])
    ax_score = fig.add_subplot(gs[0, 2])
    ax_temp  = fig.add_subplot(gs[1, 2])

    FG = "#e0e0e0"
    for ax in (ax_cur, ax_best, ax_score, ax_temp):
        ax.set_facecolor("#0f0f23")
        for sp in ax.spines.values():
            sp.set_color("#444466")
        ax.tick_params(colors=FG, labelsize=8)
        ax.title.set_color(FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)

    # ── path panels ──
    for ax, lbl in ((ax_cur,  "Current Path  [dashed line]"),
                    (ax_best, "Best Path  [solid line]")):
        ax.set_title(lbl, fontsize=11, pad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    im_cur  = ax_cur.imshow(np.zeros((n, n), np.float32),
                            cmap="plasma",  vmin=0, vmax=1,
                            aspect="auto", origin="upper", interpolation="nearest")
    im_best = ax_best.imshow(np.zeros((n, n), np.float32),
                             cmap="viridis", vmin=0, vmax=1,
                             aspect="auto", origin="upper", interpolation="nearest")

    ln_cur,  = ax_cur.plot([], [], color="white",   ls="--", lw=0.6, alpha=0.5)
    ln_best, = ax_best.plot([], [], color="#00ffcc", ls="-",  lw=0.6, alpha=0.5)

    # start (green) / end (red) markers
    mk_cs, = ax_cur.plot([], [], "go", ms=5, zorder=5)
    mk_ce, = ax_cur.plot([], [], "ro", ms=5, zorder=5)
    mk_bs, = ax_best.plot([], [], "go", ms=5, zorder=5)
    mk_be, = ax_best.plot([], [], "ro", ms=5, zorder=5)

    # ── score graph ──
    ax_score.set_title("Score", fontsize=11, pad=5)
    ax_score.set_xlabel("Elapsed (ms)", fontsize=8)
    ax_score.grid(True, alpha=0.2, color="#666688")
    ax_score.set_xlim(0, times[-1])
    ylo = float(min(cur_scores.min(), best_scores.min())) * 0.9994
    yhi = float(best_scores.max()) * 1.0006
    ax_score.set_ylim(ylo, yhi)
    ax_score.axvline(r1_ms, color="#fdcb6e", lw=0.9, ls=":", alpha=0.8)
    ax_score.axvline(r2_ms, color="#e17055", lw=0.9, ls=":", alpha=0.8)

    ln_s_best, = ax_score.plot([], [], color="#ff6b6b", ls="-",  lw=2.2,
                               label="best (solid)")
    ln_s_cur,  = ax_score.plot([], [], color="#74b9ff", ls="--", lw=1.6,
                               label="current (dashed)", alpha=0.85)
    ax_score.legend(loc="lower right", fontsize=8,
                    facecolor="#222244", labelcolor=FG, framealpha=0.8)
    vl_s = ax_score.axvline(0, color="#aaaacc", lw=0.8, alpha=0.55)

    # ── temperature graph ──
    ax_temp.set_title("Temperature", fontsize=11, pad=5)
    ax_temp.set_xlabel("Elapsed (ms)", fontsize=8)
    ax_temp.set_ylabel("T  (log scale)", fontsize=8)
    ax_temp.set_yscale("log")
    ax_temp.grid(True, alpha=0.2, color="#666688", which="both")
    ax_temp.set_xlim(0, times[-1])
    ax_temp.set_ylim(temps.min() * 0.4, temps.max() * 3)
    ax_temp.axvline(r1_ms, color="#fdcb6e", lw=0.9, ls=":", alpha=0.8)
    ax_temp.axvline(r2_ms, color="#e17055", lw=0.9, ls=":", alpha=0.8)

    ln_temp, = ax_temp.plot([], [], color="#55efc4", ls="-", lw=2.0)
    vl_t     = ax_temp.axvline(0, color="#aaaacc", lw=0.8, alpha=0.55)

    title_obj = fig.suptitle("", fontsize=12, color=FG, y=0.975)

    # ── update function ──
    def update(fi):
        elapsed, temp, cur_s, best_s, cur_p, best_p = frames[fi]
        t = times[:fi+1]

        # heatmaps
        im_cur.set_data(visit_order_img(cur_p, n))
        im_best.set_data(visit_order_img(best_p, n))

        # sparse path lines
        ln_cur.set_data(cur_p[idxs, 1], cur_p[idxs, 0])
        ln_best.set_data(best_p[idxs, 1], best_p[idxs, 0])

        # start / end markers
        mk_cs.set_data([cur_p[0,1]],   [cur_p[0,0]])
        mk_ce.set_data([cur_p[-1,1]],  [cur_p[-1,0]])
        mk_bs.set_data([best_p[0,1]],  [best_p[0,0]])
        mk_be.set_data([best_p[-1,1]], [best_p[-1,0]])

        # score lines
        cs_disp = cur_scores[:fi+1]
        bs_disp = best_scores[:fi+1]
        ln_s_best.set_data(t, bs_disp)
        ln_s_cur.set_data(t, cs_disp)

        # temperature line
        ln_temp.set_data(t, temps[:fi+1])

        # vertical time cursors
        vl_s.set_xdata([elapsed, elapsed])
        vl_t.set_xdata([elapsed, elapsed])

        title_obj.set_text(
            f"t = {elapsed:.0f} ms"
            f"   |   current = {cur_s // n2:,}"
            f"   |   best = {best_s // n2:,}"
            f"   |   T = {temp:.2e}"
        )
        return (im_cur, im_best, ln_cur, ln_best,
                mk_cs, mk_ce, mk_bs, mk_be,
                ln_s_best, ln_s_cur, ln_temp,
                vl_s, vl_t, title_obj)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=350,
        blit=False,
        repeat=True,
    )
    return fig, ani, update


# ── GIF saver (no ffmpeg needed) ─────────────────────────────────────────────

def save_gif(fig, update_fn, frames_ref, out_path, fps=4, dpi=90):
    """Render each frame to PIL Image and save as animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; install with:  pip install Pillow")
        sys.exit(1)
    import io

    nframes = len(frames_ref)
    images  = []
    print(f"Rendering {nframes} frames ...")
    for fi in range(nframes):
        update_fn(fi)
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        buf.seek(0)
        images.append(Image.open(buf).copy())
        print(f"  {fi+1}/{nframes}", end="\r", flush=True)

    print()
    hold_frames = [images[-1]] * (fps * 2)  # freeze last frame 2s
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:] + hold_frames,
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )
    print(f"Saved -> {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    test_id  = sys.argv[1] if len(sys.argv) > 1 else "0000"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[1/3] Reading input: {test_id} ...")
    n, a, in_path = read_input(test_id)

    vis_tmp = tempfile.mktemp(suffix=".bin", prefix="ahc_vis_")
    print(f"[2/3] Running solver with VIS_MODE ...")
    run_solver(in_path, vis_tmp)

    print(f"[3/3] Reading frames ...")
    frames = read_frames(vis_tmp, n)
    os.unlink(vis_tmp)

    if not frames:
        print("ERROR: no frames captured.")
        print("Make sure the solver is rebuilt after adding VIS_MODE support.")
        sys.exit(1)

    print(f"  -> {len(frames)} frames  (N={n})")

    fig, ani, update_fn = make_animation(frames, n)

    if out_path is None:
        # interactive display
        plt.show()
    elif out_path.endswith(".gif"):
        save_gif(fig, update_fn, frames, out_path, fps=4, dpi=90)
    else:
        # mp4 / etc. via ffmpeg
        fps = 4
        print(f"Saving {out_path} at {fps} fps ...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
        ani.save(out_path, writer=writer, dpi=90)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
