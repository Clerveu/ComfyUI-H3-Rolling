# ComfyUI H3 Rolling

Long video-to-video with MiniMax H3 in one node, without the picture degrading as it goes.

H3 is trained on 5-15 second clips. The usual way to go longer is to chain continuations: render a chunk, pin its last frames onto the start of the next one, repeat. That works for a few hand-offs, then contrast, sheen and blemishes start compounding, because every chunk is a copy of a copy. In our tests it got visibly bad after about 6-7 hand-offs (around 45 seconds at 5 second chunks).

This node proves out a different approach: **never chain**. Every chunk is rendered fresh and independently against its own slice of the source video, and each seam gets a short **bridge** clip pinned to the end of one chunk and the start of the next. No frame is ever more than one generation away from a fresh render, so nothing compounds, and a 70 second clip comes out looking like its first 5 seconds. You load a video, set a prompt and a reference picture, and press go. The chunking, bridging and stitching all happen inside the node.

## Status: work in progress

The degradation part is solved. The seams are mostly solved, but not completely:

- **Works well:** one subject who stays in frame, with a fairly steady camera.
- **Still rough:** the subject leaving or entering frame, several people (especially similar-looking ones), and big fast moves like a head turn that happen to land right on a seam.
- **Known H3 behavior:** H3 follows a pin at the *start* of a clip smoothly, but treats a pin at the *end* like a deadline. A bridge tends to hold the first chunk's take and then snap into the second one right before the seam. Most seams look fine. Some show a small jump.

On our 70 second test clip with structure lock on (see below), 8 of 10 seams were close to invisible and the two that weren't were head turns.

## Install

1. Clone or download this folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

The nodes show up under **MiniMax H3/rolling**.

### Requirements

- A ComfyUI recent enough to include native MiniMax H3 support. The node uses the built-in H3, sampler and audio nodes, nothing else.
- The H3 models: a Ref2VA model, the text encoder, the video VAE and the audio VAE.
- **ffmpeg on PATH** (recommended). Long renders are too big to hold in RAM as one image batch, so the node streams them to disk with ffmpeg. Without it, long renders fail.

## Nodes

### H3 Rolling Source (scrub)
Pick or upload a video, scrub the built-in player and set the start and end points. It outputs frames at 24 fps (any input frame rate is resampled, which H3 needs) and the soundtrack. `short_edge` shrinks the source: it's only choreography, and 480 is plenty for most clips. Go lower than 480 only for slow motion; fast hands and quick moves need the detail.

### H3 Rolling Ref2VA
The whole render. Minimal wiring:

```
UNET loader ---------> model
CLIP loader ---------> clip
VAE loader (video) --> vae
VAE loader (audio) --> audio_vae
H3 Rolling Source ---> frames, audio
your character image -> picture_1 (up to 4)
                       |
                 frames, audio --> Video Combine (24 fps)
```

The prompt goes to every chunk and uses the normal H3 tags: `<Picture N>` for your pictures, `<Video 1>` for the source slice, `<Audio 1>` for its soundtrack. The default prompt is a worked example (replace a skateboarder with a cat).

Settings that matter most:

| Setting | What it does |
|---|---|
| `megapixels` | Output size. Aspect comes from the source. 0.4 = 864x480 at 16:9. |
| `chunk_seconds` | Length of each independent chunk. 5 is a good default. Shorter = less VRAM, more seams. |
| `bridge_frames` | Length of each seam bridge. Longer = gentler blend between takes, more to render. |
| `seed` | In bridge mode every chunk uses the same seed. |
| `soundtrack` | `generated` (H3's audio, crossfaded at seams), `source` (the original audio untouched), or `silent`. |

The `plan` output tells you how the clip was cut up before anything renders.

**Long renders:** when the output would be too big for RAM, the full-quality file is streamed to `output/h3_rolling/rolling_<time>.mp4` and the node's `frames` output becomes a small 240p proxy. If the video your workflow saves looks low-res, that's the proxy. The real file is in `h3_rolling`.

## Optional extras

None of these are needed for the basic technique. They're all off by default.

### Structure lock
Independent chunks can disagree on framing. If the source camera slowly pushes in, each chunk under-follows it a little and the next one resets the scale, so the seams jump in size. Structure lock fixes that: SAM3 tracks the person being replaced, the node draws an edge map (Canny) of the source with that person cut out, and feeds it to the H3 Fun ControlNet as a control video. The background edges pin camera, framing and zoom to the source, and the person is still drawn fresh. On our test clip it cut the seam jumps by about 70%.

- Needs the SAM3 checkpoint in `models/checkpoints` (the node looks for `sam3.1_multiplex_fp16.safetensors`) and the H3 Fun ControlNet Union patch in `models/model_patches` (`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`).
- `lock_track`: who to track, in plain words ("man", "woman in red").
- `lock_grow`: slack around the tracked person, so the new character can be a different shape.
- `lock_max_people`: how many matches SAM3 may keep. Use 1 when there's only one person to replace.
- `lock_preview_only`: tracks and writes `output/h3_rolling/lock_preview_<time>.mp4` (tracked person in red next to the edge map), then stops. Worth checking before a long render. The console also prints a list of moments where tracking looked shaky.
- Costs about 30% slower sampling, plus a few minutes of tracking. Tracking results are cached, so re-renders of the same clip skip it.

### Upscale pass
`upscale_pass` upscales each chunk in latent space with the H3 latent upscaler and gives it a short polish at the bigger size before the next chunk is pinned to it. This needs the **Minimax H3 Latent Upscaler** node pack and its 3D upscaler model, and it only works with that upscaler, since it happens inside the render. Any normal video upscaler still works on the finished mp4 afterwards.

### Other hand-off modes
`handoff` defaults to `bridge`. The others are there for comparison: `pixel` and `latent` chain each chunk off the previous one (smooth seams, but they degrade on long clips, which is the problem this node exists to avoid), and `none` renders independent chunks and crossfades them, which looks like two takes phasing into each other.

### Extra inputs
- `sampler` / `sigmas`: plug in a workflow's own sampler and schedule instead of the widgets.
- `refmod_conditioning`: output of a RefMod Apply node, added to every chunk.

## Prompting tips

- **Don't describe camera moves in the prompt.** The same prompt goes to every chunk and H3 obeys camera wording hard, so "slow push-in" makes every chunk zoom, even after the real camera has stopped. Let the source video (and structure lock) carry the camera.
- For character swaps, describe what the new character looks like and leave the motion to `<Video 1>`.

## Related work

roycho87's [Pull-Forward](https://github.com/roycho87/comfyui-h3-pullforward) and [Clown Anchoring](https://github.com/roycho87/h3_clown_anchoring) tackle the same H3 problems from the latent side, and found the same 17-frame block structure. Clown Anchoring in particular holds real latents at the end of a clip instead of guiding toward them, which looks like a good fit for the end of a bridge. That's on the list to try.

## License

MIT
