#!/usr/bin/env python3
"""make_video.py - see the generated data on a machine without a display: GIFs and contact sheets with a tilt overlay.

    python make_video.py <episode_dir>      [--out DIR] [--every N] [--fps F] [--scale S] [--force]
    python make_video.py <dataset_dir>      [--limit N | --all] [--grid]   (same options)
    python make_video.py <anchor_XXXX.npz>  [--rollout R]
    python make_video.py <episode_XXXX.h5>

Inputs (the layouts of README.md "Data formats"):
  episode_dir   one raw episode of collect_exploration_trajs.py / collect_counterfactual_branches.py: images/000000.png ...
                (one 640 x 480 frame per state), trajectory.npz (guard_tilt_deg (T+1, 2), unsafe, grasped, time, ...) and
                metadata.json (a counterfactual branch carries outcome.counterfactual = {branch, lead, prefix_frames, ...}).
  dataset_dir   a directory with manifest.json: its first --limit complete episodes (default 3) or --all, one GIF each.
                For a counterfactual dataset, --grid puts the branches that share a source event and a lead SIDE BY SIDE
                in one GIF, frames aligned by state index ("the same moment, several futures"). A law-branch dataset
                (anchor_*.npz files) renders its first anchors as grids.
  anchor npz    collect_rrt_branches.py: pixels (R, S, 224, 224, 3), families (R,), guard_tilt_deg (R, S, 2), unsafe and
                grasped (R, S): all R rollouts as a grid of panels (4 per row), or one rollout alone with --rollout R.
  episode h5    a training-cache episode: pixels (T+1, 224, 224, 3) plus diagnostics/{guard_tilt_deg, unsafe, grasped, time}
                when present.
Outputs, in --out (default ./videos): <dataset>__<name>.gif with every --every-th state at --fps (default: every 2nd
state of a clip longer than 120 states, else every state; 10 fps, so every 2nd state of 20 Hz data plays in real time)
and, for single clips, <dataset>__<name>_strip.png with 8 evenly spaced states. The label bar above every frame (PIL's
default font): the clip name, the state index and time, the worst pillar tilt in degrees (red + UNSAFE when the recorded
unsafe flag is set), grasped yes/no, and for a counterfactual branch 'replay' before / 'BRANCH' from the branch point.
Existing files are never overwritten without --force. Only numpy, Pillow and h5py are used (no ffmpeg, no display).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONTROL_DT = 0.05                       # one action per 0.05 s (20 Hz): the clock when a clip carries no time array
LONG_CLIP = 120                         # clips with more states than this default to every 2nd state
WHITE, RED, YELLOW, GREY, BAR = (255, 255, 255), (255, 80, 80), (255, 210, 60), (170, 170, 170), (18, 18, 18)
BRANCH_ORDER = ("continue", "hold", "retreat", "push_back")       # panel order of a counterfactual grid


class Clip:
    """One image sequence with its per-state labels; frame(i) returns an (H, W, 3) uint8 array, loaded on demand."""

    def __init__(self, name, n, frame, width, tilt, unsafe, grasped=None, time=None, branch_at=None):
        self.name, self.n, self.frame, self.width = name, int(n), frame, int(width)
        # guard_tilt_deg is (states, 2) = the two pillars; the overlay shows the worse one. None -> unknown (no diagnostics).
        self.tilt = np.full(self.n, np.nan) if tilt is None else np.asarray(tilt, float).reshape(self.n, -1).max(axis=1)
        self.unsafe = np.zeros(self.n, bool) if unsafe is None else np.asarray(unsafe, bool).reshape(self.n)
        self.grasped = None if grasped is None else np.asarray(grasped, bool).reshape(self.n)
        t = np.arange(self.n) * CONTROL_DT if time is None else np.asarray(time, float).reshape(self.n)
        self.time = t - t[0]                                            # seconds since the clip's first state
        self.branch_at = branch_at                                      # counterfactual: state index of the branch point


def read_png(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def counterfactual_record(folder):
    """The branch record of a counterfactual episode (metadata.json outcome/controller.counterfactual), else None."""
    meta = read_json(folder / "metadata.json") if (folder / "metadata.json").is_file() else {}
    return (meta.get("outcome") or {}).get("counterfactual") or (meta.get("controller") or {}).get("counterfactual")


def load_episode(folder, name=None):
    """A raw episode directory: images/%06d.png per state + trajectory.npz (README 'Raw episode dataset')."""
    folder = Path(folder)
    with np.load(folder / "trajectory.npz", allow_pickle=False) as npz:
        keys = npz.files
        tilt, unsafe = npz["guard_tilt_deg"], npz["unsafe"]
        grasped, time = (npz["grasped"] if "grasped" in keys else None), (npz["time"] if "time" in keys else None)
    pngs = sorted((folder / "images").glob("*.png"))
    if len(pngs) != len(tilt):
        raise ValueError(f"{folder}: {len(pngs)} images but {len(tilt)} states in trajectory.npz")
    record, branch_at = counterfactual_record(folder), None
    if record:   # states 0..prefix_frames replay the recording; the branch law acts from the branch point on
        branch_at = int(record["prefix_frames"])
        if name is None:
            name = f"{folder.name}: {record['branch']} lead {record['lead']} (src ep {record.get('source_episode_id')})"
    if name is None:
        name = f"{folder.parent.name}/{folder.name}"
    with Image.open(pngs[0]) as first:
        width = first.size[0]
    return Clip(name, len(pngs), lambda i: read_png(pngs[i]), width, tilt, unsafe, grasped, time, branch_at)


def load_anchor(path):
    """anchor_XXXX.npz of collect_rrt_branches.py -> one Clip per rollout, named after its action-law family."""
    with np.load(path, allow_pickle=False) as npz:
        keys = npz.files
        pixels = npz["pixels"]                         # (R, S, 224, 224, 3); state 0 of every rollout = the anchor frame
        if pixels.ndim != 5 or pixels.shape[-1] != 3:
            raise ValueError(f"{path}: pixels must be (rollouts, states, H, W, 3), got {pixels.shape}")
        R, S = pixels.shape[:2]
        families = [str(f) for f in npz["families"]] if "families" in keys else []
        tilt, unsafe = npz["guard_tilt_deg"], npz["unsafe"]
        grasped = npz["grasped"] if "grasped" in keys else None
        time = npz["time"] if "time" in keys and npz["time"].shape == (R, S) else None
    clips = []
    for r in range(R):
        family = families[r] if r < len(families) else f"rollout {r}"
        clips.append(Clip(f"r{r} {family}", S, lambda i, r=r: pixels[r, i], pixels.shape[3], tilt[r], unsafe[r],
                          None if grasped is None else grasped[r], None if time is None else time[r]))
    return clips


def load_h5(path):
    """One training-cache episode (README 'HDF5 training cache'): pixels (T+1, H, W, 3) + diagnostics/* if present."""
    import h5py
    with h5py.File(path, "r") as f:
        pixels = f["pixels"][:]
        diag = f["diagnostics"] if "diagnostics" in f else {}
        values = {k: (diag[k][:] if k in diag else None) for k in ("guard_tilt_deg", "unsafe", "grasped", "time")}
    parent = path.parent.parent if path.parent.name == "episodes" else path.parent   # <cache>/episodes/<file>.h5
    return Clip(f"{parent.name}/{path.stem}", len(pixels), lambda i: pixels[i], pixels.shape[2], values["guard_tilt_deg"],
                values["unsafe"], values["grasped"], values["time"]), parent.name


# ---- drawing --------------------------------------------------------------------------------------------------------
def font_for(panel_width):
    """(font, line height) for a panel of this width: 10-16 px, about 1/24 of the width (13 px at 320, 10 px at 224)."""
    px = max(10, min(16, round(panel_width / 24)))
    try:
        return ImageFont.load_default(size=px), px + 3     # Pillow >= 10.1: scalable built-in font
    except TypeError:                                      # older Pillow: the fixed bitmap font
        return ImageFont.load_default(), 14


def panel(clip, i, scale, font, line_h):
    """Clip state i (its last state when the clip ended earlier) under a two-line label bar."""
    j = min(i, clip.n - 1)
    image = Image.fromarray(clip.frame(j))
    if scale != 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    bar = 2 * line_h + 4
    out = Image.new("RGB", (image.width, image.height + bar), BAR)
    out.paste(image, (0, bar))
    draw = ImageDraw.Draw(out)
    draw.text((3, 1), clip.name, fill=WHITE, font=font)
    parts = [(f"#{j} {clip.time[j]:.2f}s ", WHITE)]
    if np.isfinite(clip.tilt[j]):
        parts.append((f"tilt {clip.tilt[j]:.1f} deg" + (" UNSAFE" if clip.unsafe[j] else ""), RED if clip.unsafe[j] else WHITE))
    else:
        parts.append(("tilt ?", GREY))
    if clip.grasped is not None:
        parts.append((" grasp " + ("yes" if clip.grasped[j] else "no"), WHITE))
    if clip.branch_at is not None:
        parts.append((" BRANCH", YELLOW) if j >= clip.branch_at else (" replay", GREY))
    if i > j:
        parts.append((" end", GREY))
    x = 3.0
    for text, colour in parts:
        draw.text((x, line_h + 2), text, fill=colour, font=font)
        x += font.getlength(text)
    return out


def compose(clips, i, scale, font, line_h, cols):
    """All clips at state i as a grid of panels, cols per row (one clip -> just its panel)."""
    panels = [panel(c, i, scale, font, line_h) for c in clips]
    if len(panels) == 1:
        return panels[0]
    w, h = max(p.width for p in panels), max(p.height for p in panels)
    sheet = Image.new("RGB", (cols * w, math.ceil(len(panels) / cols) * h), BAR)
    for k, p in enumerate(panels):
        sheet.paste(p, ((k % cols) * w, (k // cols) * h))
    return sheet


def shared_palette(samples):
    """One 255-colour palette for the whole GIF (Pillow's median cut on a column of sample frames). With one palette,
    unchanged pixels map to the same index in every frame, so Pillow stores only the changed box of each frame; the
    256th index stays free for the transparency trick of optimize=True. 255 colours suit the flat-shaded scene."""
    mosaic = Image.new("RGB", (samples[0].width, samples[0].height * len(samples)))
    for k, s in enumerate(samples):
        mosaic.paste(s, (0, k * samples[0].height))
    return mosaic.quantize(colors=255)


def write_gif(path, clips, indices, cols, scale, fps):
    font, line_h = font_for(clips[0].width * scale)
    sample = [compose(clips, i, scale, font, line_h, cols) for i in indices[:: max(1, len(indices) // 12)][:12]]
    palette = shared_palette(sample)
    frames = [compose(clips, i, scale, font, line_h, cols).quantize(palette=palette, dither=Image.Dither.NONE) for i in indices]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return len(frames), frames[0].size


def write_strip(path, clips, n, cols, scale):
    """Contact sheet: 8 evenly spaced states, 4 per row, each with the same label bar as the GIF."""
    font, line_h = font_for(clips[0].width * scale)
    states = sorted({int(round(x)) for x in np.linspace(0, n - 1, 8)})
    tiles = [compose(clips, i, scale, font, line_h, cols) for i in states]
    w, h = tiles[0].size
    sheet = Image.new("RGB", (4 * w, math.ceil(len(tiles) / 4) * h), BAR)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 4) * w, (k // 4) * h))
    sheet.save(path)
    return len(tiles), sheet.size


# ---- what to render -----------------------------------------------------------------------------------------------
def plan_jobs(path, args):
    """-> list of {stem, clips, cols, strip}; raises ValueError for an input that is not one of the four kinds."""
    path = Path(path).resolve()                 # output names use the parent directory's name, so 'episode_0000' alone works
    if path.is_dir() and (path / "manifest.json").is_file():
        return dataset_jobs(path, read_json(path / "manifest.json"), args)
    if path.is_dir() and (path / "trajectory.npz").is_file() and (path / "images").is_dir():
        return [{"stem": f"{path.parent.name}__{path.name}", "clips": [load_episode(path)], "cols": 1, "strip": True}]
    if path.is_file() and path.suffix == ".npz":
        return [anchor_job(path, path.parent.name, args.rollout)]
    if path.is_file() and path.suffix in (".h5", ".hdf5"):
        clip, cache = load_h5(path)
        return [{"stem": f"{cache}__{path.stem}", "clips": [clip], "cols": 1, "strip": True}]
    raise ValueError(f"{path}: not an episode directory (images/ + trajectory.npz), a dataset directory (manifest.json), "
                     "an anchor_XXXX.npz or a training-cache episode .h5")


def anchor_job(path, dataset, rollout):
    clips = load_anchor(path)
    if rollout is None:
        return {"stem": f"{dataset}__{path.stem}", "clips": clips, "cols": min(4, len(clips)), "strip": False}
    if not 0 <= rollout < len(clips):
        raise ValueError(f"{path}: --rollout {rollout} out of range (rollouts 0..{len(clips) - 1})")
    clips[rollout].name = f"{path.stem} {clips[rollout].name}"
    return {"stem": f"{dataset}__{path.stem}_r{rollout}", "clips": [clips[rollout]], "cols": 1, "strip": True}


def dataset_jobs(folder, manifest, args):
    limit = None if args.all else args.limit
    anchors = sorted(folder.glob("anchor_*.npz"))
    if anchors or "anchors" in manifest:                                   # a law-branch dataset (rrt_branches_v1)
        if args.grid:
            raise ValueError(f"{folder}: --grid is for a counterfactual dataset; anchors are always rendered as grids")
        if not anchors:
            raise ValueError(f"{folder}: a branch dataset without anchor_*.npz files")
        return [anchor_job(p, folder.name, args.rollout) for p in anchors[:limit]]
    rows = [r for r in manifest.get("episodes", []) if r.get("status", "complete") == "complete"]
    if not rows:
        raise ValueError(f"{folder}: manifest.json lists no complete episode")
    if not args.grid:
        return [{"stem": f"{folder.name}__{r['path']}", "clips": [load_episode(folder / r["path"])], "cols": 1, "strip": True}
                for r in rows[:limit]]
    # --grid: the branch episodes of one (source event, lead) share the same replayed prefix and diverge at the branch point
    if manifest.get("interaction_revision") != 4 and "counterfactual_branches" not in manifest:
        raise ValueError(f"{folder}: --grid needs a counterfactual dataset (interaction_revision 4); this manifest is not one")
    groups = {}
    for r in rows:
        rec = counterfactual_record(folder / r["path"])
        if rec is None:
            raise ValueError(f"{folder / r['path']}: metadata.json has no counterfactual record")
        groups.setdefault((int(rec["source_dataset"]), int(rec["source_episode_id"]), int(rec["lead"])), []).append((rec["branch"], r["path"]))
    jobs = []
    for (ds, ep, lead), members in list(groups.items())[:limit]:
        members.sort(key=lambda m: (BRANCH_ORDER.index(m[0]) if m[0] in BRANCH_ORDER else len(BRANCH_ORDER), m[0]))
        clips = [load_episode(folder / p, name=f"{branch}  lead {lead}") for branch, p in members]
        jobs.append({"stem": f"{folder.name}__src{ds}-{ep:04d}_lead{lead:02d}_grid", "clips": clips, "cols": len(clips), "strip": False})
    return jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="episode directory | dataset directory (manifest.json) | anchor_XXXX.npz | episode .h5")
    p.add_argument("--out", type=Path, default=Path("videos"), help="output directory (default ./videos, created if needed)")
    p.add_argument("--every", type=int, default=0, help="render every N-th state (default 0 = 2 for clips longer than 120 states, else 1; the last state is always included)")
    p.add_argument("--fps", type=float, default=10.0, help="GIF frames per second (default 10)")
    p.add_argument("--scale", type=float, default=0.0, help="frame scale (default 0 = 0.5 for raw 640 x 480 frames, 1 for 224 px cache frames)")
    p.add_argument("--limit", type=int, default=3, help="dataset directory: episodes / grids / anchors to render (default 3)")
    p.add_argument("--all", action="store_true", help="dataset directory: render all of them")
    p.add_argument("--grid", action="store_true", help="counterfactual dataset: one GIF per (source event, lead) with its branches side by side")
    p.add_argument("--rollout", type=int, default=None, help="anchor npz: render this rollout alone instead of the grid of all rollouts")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    a = p.parse_args(argv)
    if a.every < 0 or a.fps <= 0 or a.scale < 0 or a.limit < 1:
        print("FAIL make_video.py: --every, --limit must be positive, --fps > 0, --scale >= 0")
        return 2
    try:
        jobs = plan_jobs(a.path, a)
    except (ValueError, OSError, KeyError) as e:
        print(f"FAIL make_video.py: {e}")
        return 2
    outputs = [a.out / (job["stem"] + ext) for job in jobs for ext in ([".gif", "_strip.png"] if job["strip"] else [".gif"])]
    existing = [o for o in outputs if o.exists()]
    if existing and not a.force:
        print(f"FAIL make_video.py: {existing[0]} exists (use --force to overwrite, or another --out)")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        clips, n = job["clips"], max(c.n for c in job["clips"])
        every = a.every or (2 if n > LONG_CLIP else 1)
        scale = a.scale or (0.5 if clips[0].width > 400 else 1.0)
        indices = list(range(0, n, every)) + ([n - 1] if (n - 1) % every else [])
        gif = a.out / (job["stem"] + ".gif")
        frames, size = write_gif(gif, clips, indices, job["cols"], scale, a.fps)
        print(f"wrote {gif}  ({gif.stat().st_size / 1048576:.2f} MB, {frames} frames of {n} states, {size[0]}x{size[1]}, {a.fps:g} fps)")
        if job["strip"]:
            strip = a.out / (job["stem"] + "_strip.png")
            tiles, size = write_strip(strip, clips, n, job["cols"], scale)
            print(f"wrote {strip}  ({strip.stat().st_size // 1024} kB, {tiles} states, {size[0]}x{size[1]})")
    print(f"{len(outputs)} files in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
