# Plan: 48 trials per song (calibration + runs 1–4)

Status: implemented 2026-10-20, not yet compiled or played in Unity. `unity_noVR` only.

## Goal
- Calibration and runs 1–4 use the earlier Ori song set, with 48 trials each (was 40).
- Keep the trial timing of the 40-trial maps.

| Role | Song asset (`Assets/SO/Songs/`) | Folder (`StreamingAssets/`) |
|---|---|---|
| Calibration | `ShriekAndOri` | `Calibration` |
| Run 1 | `ClimbingTheGinsoTree` | `run1` |
| Run 2 | `NowUseTheLight` | `run2` |
| Run 3 | `LumaPools` | `run3` |
| Run 4 | `OriAndTheBlindForest` | `run4` |

All five assets were already referenced by `Lobby.unity`.

## How the trial count is set
- The map sets it, not a config value. With `use_random: 1` (the MI scenes), `BeatmapSpawner.set_random_old_note` groups the notes with `GroupNotesByRules`. Each group (`group_note_num` = 5 notes) is one trial.
- There is no cap. The odd-group trim is commented out. Unity logs `Group num is N` at song start.
- The run ends when the last note spawns (`BeatmapSpawner.Update`, `currentNoteIndex >= old_notes.Count` → `GM.SetFinalAcc()`), not when the audio ends.
- "MI trial: 48" in `SongSelectMenu.get_note_num` is a fixed label, not a count.

## Measured (40-trial maps, before the change)
Times are from the `.dat` note times and BPM in `Info.dat`. Audio length is read from the `.ogg` headers.

| Song | Notes | Last note | Audio |
|---|---|---|---|
| ShriekAndOri | 200 | 283 s | 300 s |
| ClimbingTheGinsoTree | 200 | 291 s | 301 s |
| NowUseTheLight | 200 | 296 s | 301 s |
| LumaPools | 200 | 287 s | 300 s |
| OriAndTheBlindForest | 200 | 298 s | 301 s |

**Trial timing** (run1; ShriekAndOri is similar):
- 5 notes about 0.7–0.8 s apart, so the 5 notes span about 3.0 s.
- About 3.7 s from the last note of a trial to the next trial's first note.
- Start to start: median 6.8 s, range 6.6–12.0 s.

**What this means:**
- 48 trials at this spacing need about 340–355 s, which is longer than the 5 min audio.
- **Decision:** keep the spacing (the "keep 7 s" option). The music loops instead of shrinking the rest period.
  - Shrinking the rest period to about 2.5 s would fit in 5 min, but it changes the protocol.
  - Python timing values such as `vote_offset_s` and `swap_guard_s` were tuned on the current timing.

## Changes
1. **New maps:** [`python/else/music_tools/extend_maps_to_48.py`](../python/else/music_tools/extend_maps_to_48.py).
   - Reads each 40-trial map and appends 8 groups of 5 notes.
   - Each new group starts one median start-to-start period after the previous group. Its in-trial offsets are copied from groups 1–8.
   - Writes `<name>_48.dat` beside the original. The originals are not modified.
   - Left/right is re-shuffled at runtime, so the copied `_lineIndex`/`_type` don't matter.
   - Re-counts the trials with a Python port of `GroupNotesByRules` and exits non-zero unless it finds 48 groups of 5.
   - Result: every map has 240 notes and 48 trials. Median start-to-start is 6.8–7.0 s. The last note lands at 339 / 345 / 351 / 343 / 354 s.
2. **Song assets:** `file_names[0]` in the five assets now points at the `_48.dat` file.
3. **Audio loop:** `BeatSaberInfoLoader.cs` has a new `public bool loop_audio = true;`, and `audioSource.loop = loop_audio;` is set before playback. Pause and continue are unchanged.
4. **Label:** `SongSelectMenu.cs` line 154 already reads `"MI trial: 48";` in the working tree. It was edited outside this work, and git shows 29 changed lines there that look like a line-ending or encoding change. Check it before committing.

## Verification (open)
- [ ] Unity reimports. New `.dat.meta` files appear, and the `.asset` edits aren't overwritten by an open Editor.
- [ ] It compiles.
- [ ] Each of the five songs logs `Group num is 48`.
- [ ] The run ends after trial 48. Python sees trials 0–47 per run.
- [ ] Listen to the loop point at about 300 s. The songs are 5 min cuts, so the jump back to the start may be audible. If it is, make a longer audio file with a crossfade (needs ffmpeg, not checked).
- [ ] `git status` shows only the intended files.

## Not checked
- Whether any Python code assumes 40 trials per run. The docs already mention 48 per run.
- `Lobby.unity` also has a `BeatSaberInfoLoader`, so it gets `loop_audio = true` by default. I didn't check whether it plays audio there.
- The `unity/` (VR) project is not changed.

## Rerun
`python python/else/music_tools/extend_maps_to_48.py` overwrites the `_48.dat` files and prints one OK or FAIL line per song.
