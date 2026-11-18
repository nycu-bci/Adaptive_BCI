# python python/else/music_tools/extend_maps_to_48.py
# Extends the 40-trial augmented MI maps to 48 trials (5 notes per trial) and checks the trial count
# with a port of BeatmapSpawner.GroupNotesByRules. Writes new files, never overwrites the originals.
import copy
import json
import statistics
import sys
from pathlib import Path

STREAMING = Path(__file__).resolve().parents[3] / "unity_noVR" / "Assets" / "StreamingAssets"

# folder -> source map (the one the Song asset uses now)
MAPS = {
    "Calibration": "MI_ExpertPlusStandard_v2_augmented_65s_new.dat",
    "run1": "MI_ExpertStandard_augmented_65s_new.dat",
    "run2": "MI_ExpertPlusStandard_augmented_65s_new.dat",
    "run3": "MI_ExpertPlusStandard_augmented_65s_new.dat",
    "run4": "MI_ExpertStandard_augmented_65s.dat",
}

TARGET_TRIALS = 48
NOTES_PER_TRIAL = 5      # Config.group_note_num
CUBE_SPACE_TIME = 0.7    # Config.cube_space_time


def load(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def group_count(times, bps):
    """Port of BeatmapSpawner.GroupNotesByRules (n = group_note_num)."""
    n = NOTES_PER_TRIAL
    up = (CUBE_SPACE_TIME * n + 2) * bps
    down = CUBE_SPACE_TIME * n * bps
    one = bps
    groups, i, total = [], 0, len(times)
    while i < total:
        g = [times[i]]
        look = 1
        while look < n and i + look < total:
            if times[i + look] - times[i] > up:
                break
            g.append(times[i + look])
            look += 1
        if len(g) < n and g[-1] - g[0] < down:
            i += len(g)
            continue
        groups.append(g)
        skip = len(g) + 1
        nxt = i + len(g)
        if nxt < total and times[nxt] - g[-1] > one:
            skip = len(g)
        i += skip
    return groups


def extend(folder, name):
    src = STREAMING / folder / name
    info = load(STREAMING / folder / "Info.dat")
    bps = info["_beatsPerMinute"] / 60.0
    data = load(src)
    notes = sorted(data["_notes"], key=lambda x: x["_time"])
    if len(notes) % NOTES_PER_TRIAL:
        sys.exit(f"{src}: {len(notes)} notes is not a multiple of {NOTES_PER_TRIAL}")
    groups = [notes[i:i + NOTES_PER_TRIAL] for i in range(0, len(notes), NOTES_PER_TRIAL)]
    have = len(groups)
    if have >= TARGET_TRIALS:
        sys.exit(f"{src}: already {have} groups")

    # repeat the trial-to-trial period (median start-to-start, in beats) and the first trials' in-trial offsets
    period = statistics.median(b[0]["_time"] - a[0]["_time"] for a, b in zip(groups, groups[1:]))
    out = [dict(n) for n in notes]
    last_start = groups[-1][0]["_time"]
    for k in range(TARGET_TRIALS - have):
        template = groups[k % have]
        base = template[0]["_time"]
        last_start += period
        for n in template:
            new = copy.deepcopy(n)
            new["_time"] = round(last_start + (n["_time"] - base), 4)
            out.append(new)
    data["_notes"] = out

    dst = src.with_name(src.stem + "_48.dat")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    times = sorted(n["_time"] for n in out)
    g = group_count(times, bps)
    sizes = {len(x) for x in g}
    starts = [x[0] * 60 / info["_beatsPerMinute"] for x in g]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    ok = len(g) == TARGET_TRIALS and sizes == {NOTES_PER_TRIAL}
    print(f"{folder:12s} {name} -> {dst.name}: {len(out)} notes, {len(g)} trials, group sizes {sorted(sizes)}, "
          f"median start-to-start {statistics.median(gaps):.1f}s, ends {times[-1] * 60 / info['_beatsPerMinute']:.0f}s "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    results = [extend(folder, name) for folder, name in MAPS.items()]
    sys.exit(0 if all(results) else 1)
