# video/

Rendered by `scripts/video.sh <iteration>` (src/render.py) from `iterations/<iteration>/clips.json`.

```
video/<iteration>/e<epoch>_<cell>_s<seed>[_<tag>]/
  INDEX.md          start here: every episode of the rollout (failure mode, event frames) and every clip with
                    what to look for
  clips/<id>__env<NN>__f<a>-<b>.mp4   requested moments, slow motion (0.5x)      (git-ignored)
  full/env<NN>.mp4                    whole episodes, real time                  (git-ignored)
  sheets/<id>.png                     contact sheet: 5 frames around each clip's anchor (committed)
  events.json, trace.npz              this rollout's failure events and per-frame trace
```

Reading a frame: left = wide fixed view across the desk, right = close-up following the book. Red box = the
simulated book, **green wireframe = where the reference book is now**, green dots = reference head / elbows /
knees / ankles / fingertips. HUD: frame t (0..147) and reference frame, phase, book error / height / distance vs
the reference, wrist distances, fingertip contact force, orientation error vs the reference, status (IN HAND /
FREE / ON FLOOR / FALLEN, TRACK OK / LOST), and a timeline with the reference phases, this episode's events and the
current frame (magenta).

The video rollout is a fresh run of the same policy (GPU physics is not bit-reproducible), so its episodes are not
the evaluation's episodes; the clips are cut at the events of this rollout.
