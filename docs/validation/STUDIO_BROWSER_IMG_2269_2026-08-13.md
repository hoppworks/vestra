# Vestra Studio browser validation — IMG_2269

Date: 2026-08-13  
Scope: local browser rendering of the existing private `IMG_2269` reconstruction bundle.  
Scale: relative; this is not a metric accuracy claim.

## Environment

- Studio source: Vestra root checkout, served by `vestra-cli serve` on a loopback-only address.
- Scene: `/var/roothome/vestra-runs/img-2269-7e3a334.vestra` on the Workhorse.
- Browser access: SSH loopback forwarding to `http://127.0.0.1:14318`.
- Automation: `agent-browser`; no external upload or hosted service was used.

## Observed scene evidence

The served fused layer reported:

- 300,906 fused surfels
- 14 reconstruction windows
- 7 progressive fused segments
- 13 sequential seams
- 88.1% minimum sequential inlier ratio
- 0 accepted loop closures
- relative scale and `ready` capture disposition

## Interaction contract exercised

1. Open the fused world.
2. Enable camera rays.
3. Switch to measured evidence.
4. Switch back to the fused world.

After both layer switches the browser reported the expected button states: `SHOW MEASURED EVIDENCE` and `HIDE CAMERA RAYS`. The final screenshot visually showed the fused world and its camera rays. This validates that camera-ray WebGL buffers are rebuilt from the current layer extent rather than retaining stale geometry or accumulating buffers across switches.

The same browser run loaded the newly added local source picture-in-picture,
then advanced from source frame `001 / 120` to `002 / 120`. The image was
served from the bundle's existing decoded RGB24 cache through a numeric,
loopback-only route; no source imagery was uploaded or committed.

The calibrated camera overlay was then enabled for the same bundle. Studio
reported `HIDE CAMERA RAYS` and rendered the camera directions together with
their intrinsic image-plane frustums. The source panel was placed independently
of the ledger so it did not obstruct either diagnostic control.

## Commands

```sh
# On the Workhorse
cargo run -q -p vestra-cli -- serve \
  --scene /var/roothome/vestra-runs/img-2269-7e3a334.vestra \
  --port 4318

# On the local machine
ssh -fN -L 14318:127.0.0.1:4318 workhorse
npx agent-browser open http://127.0.0.1:14318
```

The bundle remains private. No screenshot or video frame is checked into this repository.
