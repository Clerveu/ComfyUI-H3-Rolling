# ComfyUI H3 Rolling

Two nodes for long MiniMax H3 videos that never copy a render forward more than once:

- **H3 Rolling Timeline**: a long video from pictures and words. You type a storyboard, it renders the whole thing.
- **H3 Rolling Ref2VA**: the same idea driven by a source video (character swaps and the like).

## Read this first

This is a personal tool, shared for reference. I built it to make my own videos, I'm not maintaining it as a product, and I can't promise support. Take whatever is useful.

**The core idea here is not new, and I found that out late.** Rendering fresh clips and then filling the gap between two of them with a clip that is locked to both is:

- **Bridge** in chanon's [comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline), which is a full timeline editor, and
- the "odd shots fresh, even shots continue" example (`withers_has_big_nattys`) in roycho87's [Geiru Context Suite](https://github.com/roycho87/geiru_context_suite), using his Geiru Context and Geiru Anchoring nodes. Roy credits the odd/even order to obvpm.

Holding real latents under a mask so the sampler keeps them comes from drozbay ([MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments)) and seitanism ([Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)); I learned it from Roy's nodes. No code from any of those projects is in this pack, but most of the ideas are theirs. I worked on this in parallel without realising how much of it they had already covered, so treat this repo as one more set of notes on the same road, not a different road.

**If you're choosing a tool:**

- Want to build a video clip by clip, look at each take, re-roll, trim, loop, upscale the whole cut in one pass? Use chanon's timeline. It does far more than this.
- Want long chains where motion carries forward naturally from clip to clip, or one long render in windows? Use Roy's suite. His Keystone work holds a chain together well, and a chain keeps pace between clips better than this does. What you give up is how fast you can change things. In a chain every clip is built on the one before it, so re-doing clip 4 means re-doing everything after it. Here a changed shot costs that shot and the two bridges touching it, and everything else comes back from a cache. The more control you want over individual moments, the more that matters.
- Want to type a whole storyboard into one box, press Run once and walk away, with each piece only ever one step from a fresh render? That is the one thing this node is for.

## What might be worth a look

Small things I didn't find written down elsewhere. They may well be known.

- **Changing the room through a bridge.** Shots render first, so a bridge is locked at its start to frames that show the room before a change and at its end to frames that show it after. Give the bridge the *old* reference picture and only give the new picture to the shot after it. In my test a character writes on a blank chalkboard and her writing arrives at the handwriting that is in the next shot's reference picture. See the worked example below.
- **A short shot between two bridges.** One long action (writing two lines) split into two bridges with a 1.5 second shot between them, whose only job is the motion between the halves. It reads as a breath, not a pause.
- **A number behind landing 17 frames in.** Roy's recommended Geiru Anchoring setting frees the first 17 frames of the next shot (`drop_first 17`), and his notes say of the drop settings that freeing the outer 17 frames "lets motion carry instead of freezing." I ran into the same thing from the other side on a running subject. Across the 15 joins of one run, the first frame-to-frame step of a freshly rendered shot was often a held frame, moving as little as 0.08 of the normal amount, sometimes with a second one at frames 5 to 6. A bridge arriving on those frames slows into them and then jumps. Arriving 17 frames later fixed it. This node always does that (it renders every shot after the first with 17 extra frames at the start, arrives after them, and never shows them).

## Clips

Most of these have a label burned into the corner: which piece each frame belongs to (shot or bridge), the frame number inside it, and the frames a bridge shares with a shot in yellow.

### What it does now

**A storyboard: writing on a chalkboard** (3 shots, 2 bridges, 35 seconds, upscaled). The room's reference picture changes twice, each time through a bridge. This is the worked example further down.

https://github.com/user-attachments/assets/971a092f-26b7-40ec-8fc7-7471c91a90d3

**Continuous movement: a run, 10 seconds per piece, bridges holding 22 frames** (4 shots, 3 bridges, 63 seconds, one prompt, one character sheet, no motion reference). This is how I'd set up long-form movement now.

[![10 second pieces with 22-frame holds](https://img.youtube.com/vi/tmJ6Kr12E8k/hqdefault.jpg)](https://youtube.com/shorts/tmJ6Kr12E8k)

(Opens on YouTube.)

### Things you can choose between

**How much a bridge holds: 5 frames, then 22.** The same four shots of a run. First clip: bridges holding 5 frames. Second clip: only the bridges re-rendered, each holding 22 frames and re-making the last 17 of the shot before it. Shots 1, 2 and 4 are the same footage in both. Shot 3 was re-rolled for the second clip, because the original ran slow on its own from its first frame. For continuous movement like this, 22 is the one to use. 5 is still the better fit for short storyboard transitions; see "How much a bridge holds".

https://github.com/user-attachments/assets/d417e770-16a7-4404-bd67-5b880bb7190f

https://github.com/user-attachments/assets/9b338e8d-2421-4cb1-b367-2f8355f78d0e

### Earlier attempts, and why they were replaced

**Arriving on a shot's first frame.** Two joins, each shown both ways (arriving on the first frame, then arriving 17 frames in), at normal speed and then at quarter speed. Arriving on the first frame jumped on about a third of joins with a moving subject. The node now always arrives 17 frames in, so the first version of each join can no longer happen.

https://github.com/user-attachments/assets/08e3837a-ae41-49ad-9039-d3365375d48c

https://github.com/user-attachments/assets/aa347f72-8aef-4267-af25-41d05994cde3

**A long run WITHOUT the 22-frame holds: 16 shots and 15 bridges of 5 seconds each, one prompt, 2 minutes 34 seconds.** The picture at the end looks like the picture at the start, which is what it was made to show. But **her pace changes from piece to piece, and sometimes drifts inside one.** Watch the labels and you'll see the speed shift as the pieces change. This is what continuous movement does with 5-frame holds and short pieces. Holding 22 frames mostly fixes it (the clips above), so this is no longer how I'd make a run. It stays up as the example of what happens if you don't.

[![Long run, preview size, with labels](https://img.youtube.com/vi/4NjcilVhwvI/hqdefault.jpg)](https://youtube.com/shorts/4NjcilVhwvI)
[![The same run through the upscale pass, no labels](https://img.youtube.com/vi/zM0mxxwC2h8/hqdefault.jpg)](https://youtube.com/shorts/zM0mxxwC2h8)

Left: preview size, with labels. Right: the same run through the upscale pass, no labels. (Both open on YouTube.)

**A motion reference with 5-frame holds** (5 pieces, 25 seconds). Before the 22-frame holds, the way to steady the pace was a short looping clip of a plain mannequin running, given to every piece as `<Video 1>`. It took most of the pace change out, not all of it, and a reference video makes every piece slower to render. The 22-frame holds do more of this job for less.

https://github.com/user-attachments/assets/4fde8d3b-f764-4930-b164-ce77bb65b806

## Install

1. Clone or download this folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

The nodes show up under **MiniMax H3/rolling**. A starter workflow is in `workflows/H3_Rolling_Timeline.json`.

### Requirements

- A ComfyUI recent enough to include native MiniMax H3 support.
- The H3 models: a Ref2VA model, the text encoder, the video VAE and the audio VAE.
- **ffmpeg on PATH** (recommended). Long renders are too big to hold in RAM, so the node streams them to disk with ffmpeg.
- Optional: the [Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) pack for `upscale_pass`.

Nothing else is required. Roy's and chanon's packs are not dependencies.

### What it was tested with

Everything in this README was made on one setup. I have not tried it any other way, so treat anything outside this list as untested.

| | |
|---|---|
| Model | `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8` (a Ref2VA model) |
| Speed LoRA | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16` at strength 1.0, on every piece |
| Sampling | er_sde, beta schedule, 6 steps. These are the node's built-in values when `sampler` and `sigmas` aren't wired. |
| Attention | Comfy Kitchen attention (Model Attention Backend node) |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq` |
| VAEs | `minimax_h3_video_vae_int8_convrot`, `minimax_h3_audio_vae_fp32` |
| Size | 864x480 (or 480x864) for the first stage |
| Upscale pass | `minimax_h3_latent_upscaler_3d_conv_v1_fp16` to 1.5 MP, then 2 steps at 0.4 denoise |
| Reference pictures | `ref_image_size` = max |
| Software, hardware | ComfyUI 0.42.5, RTX 5090 |

**It was only ever run with the turbo LoRA.** Roy's README says the choice of turbo changes how H3 treats held frames, so a different turbo, or none, may behave differently at the joins. If you run without a turbo, wire your own `sampler` and `sigmas`; 6 steps is far too few without one.

For scale: the 35 second chalkboard example (3 shots, 2 bridges) took about 7 minutes at 864x480, and about 12 more for the upscale pass.

---

# H3 Rolling Timeline

## How it works

The video is a list of pieces that alternate: shot, bridge, shot, bridge, shot.

- A **shot** is a normal H3 render from your pictures and its words.
- A **bridge** is a normal H3 render whose first frames are frames of the shot before it and whose last frames are frames of the shot after it: 5 at each end by default, 22 if you ask for it (see "How much a bridge holds"). Those frames are the shots' real latents, written in and masked, so they decode as very nearly the same frames (within a couple of brightness levels in my test) and the joins are plain cuts. Where the two overlap, the shot's own frames are the ones shown.
- All the shots render first. Then the bridges. Nothing is ever rendered from a bridge, so no frame is more than one step from a fresh render and nothing compounds, however long the video is.
- Inside a bridge, at the frame where its own generated frames meet the locked ones, there was a small one-frame brightness step: 2.95 levels out of 255 at the arriving end, measured on one join. The node eases each bridge's free frames onto the shots at both ends with a colour curve, which took that step to 0.6.

Lengths are H3's 17k+5 frames (22, 39, 56 ... up to 15 seconds). Shot N uses `seed + N - 1`, so the same prompt gives different shots.

## The simplest use

Put a prompt in **prompt**, plug in your pictures, set **shots**. Every piece gets the prompt exactly as typed, with every picture. `<Picture 1>` is `picture_1`, and so on. That's all.

There are nine picture inputs (H3's limit) and they take anything that outputs an image: plain Load Image nodes, as in the starter workflow, or any media loader. Empty inputs are ignored, and the input's number is the picture's number even if lower ones are empty.

## The storyboard box

Optional. Everything in it is optional too: leave a part out and nothing is added for it.

`>` starts a shot and `~` starts a bridge. Whatever follows, up to the next marker, is that piece's text. The text goes where `{beat}` is in the prompt. If the prompt has no `{beat}` it goes on the end, and if the prompt box is empty it is the whole prompt.

```
> She stands at the lectern and looks at the camera.
~ She walks to the window.
> She stands at the window, looking out.
```

Settings are bare lowercase words right after the marker:

| Setting | What it does |
|---|---|
| `7s` or `158f` | This piece's length. Seconds snap to the nearest length H3 can make. Without it, the piece uses `shot_seconds` / `bridge_seconds`. |
| `image 3` | Start handing `picture_3` to this piece and every piece after it. |
| `-image 2` | Stop handing `picture_2` over, from this piece on. |
| `take 2` | Another roll of this piece (a different seed). |
| `seed 5001` | This exact seed for this piece. |
| `solo` | Render only this piece. A bridge brings its two shots. |
| `hold 5`, `hold 22`, `hold 22 remake` | On a bridge line only: how much of its two shots this one bridge holds. See "How much a bridge holds". |

Above the first piece you can describe the pictures, and the node will write the `subject_definitions` and `retention_analysis` lines for each piece from the pictures that piece actually has, with the picture numbers filled in:

```
image 1: <Subject 1> is the woman from the multi-view sheet <Picture>.
  keep: her appearance and clothing from <Picture>
image 2, 3, 4: <Subject 2> is the lecture room in <Picture>.
  keep: the room in <Picture>
start: image 1, image 2
```

- `<Picture>` with no number is filled in per piece. H3 numbers pictures in the order it receives them, so image 4 is `<Picture 2>` in a piece that only has images 1 and 4.
- `keep:` is optional (left off, the line says "as shown in `<Picture N>`"). `start:` is optional (left off, every loaded picture).
- The lines go under `subject_definitions:` and `retention_analysis:` if your prompt has those headings, and are added at the top if it doesn't.
- `<Picture N>` typed by hand always means `picture_N` and is renumbered for you. If a piece mentions a picture it isn't being handed, the run continues and the **plan** output says so.

### Worked example: writing on a chalkboard

Four pictures: a character sheet, and the same room three times (blank board, one line written, both lines written).

prompt:

```
subject_definitions:

summary:
[reference generation] {beat}

retention_analysis:

detailed_description:
Photorealistic live-action footage, filmed by a static camera.

[Shot 1] A static, locked-off, straight-on view of the front wall of <Subject 2>, showing the chalkboard, the lectern and the projector screen, with <Subject 1> standing behind the lectern. {beat}

overall_soundscape:
Quiet room tone.

non_diegetic_music:
N/A
```

storyboard (after the `image` lines above):

```
> 6.5s <Subject 1> (S1) stands facing the camera with both hands on the top of the lectern and says, <d>[English] Blackboard test, take one.</d> She then turns around to face the clean, empty chalkboard, picks up a stick of chalk from the tray and holds it up near the top left of the chalkboard.
~ 11s <Subject 1> stands facing the chalkboard with her back to the camera and writes "LATENT SPACE" with the stick of chalk in large capital letters across the upper half of the chalkboard.
> 1.5s image 3 -image 2 <Subject 1> stands facing the chalkboard with her back to the camera, lifts the stick of chalk off the chalkboard and lowers her hand to the left side of the chalkboard, below the line of writing, holding the chalk just off the surface.
~ 11.5s <Subject 1> stands facing the chalkboard with her back to the camera and writes "IS A TRAP!" with the stick of chalk in large capital letters across the chalkboard, below the line already written.
> 5s image 4 -image 3 <Subject 1> puts the stick of chalk back in the tray, turns around to face the camera and rests both hands on the top of the lectern.
```

The first bridge still has the blank-board picture, but it ends locked to the next shot, which was rendered from the one-line picture. So she writes the line.

## Changing one piece

Every rendered piece is cached under everything that went into it. A shot depends on its own words, length, pictures and seed. A bridge depends on those plus its two shots. Change one shot and re-run: that shot and the two bridges touching it render again, everything else comes back from the cache. The last line of the **plan** output says what was rendered and what was reused.

`solo` uses the same cache. Trial a shot on its own with `solo` and `take 2`, `take 3` ... until you like it, then remove `solo`; the full run picks the shot up without rendering it again. `upscale_pass` also reuses pieces that were already rendered at preview size.

The cache lives in `piece_cache/` inside the node's folder and is capped at 8 GB.

## How much a bridge holds

The `bridge_hold` setting says how much of its two neighbours every bridge holds:

| `bridge_hold` | What a bridge does |
|---|---|
| `5 frames` (default) | Holds the last 5 frames of the shot before it and 5 frames of the shot after it. |
| `22 frames` | Holds 22 at each end. |
| `22 frames, re-make the shot's last 17` | Holds 22 at each end, and starts 17 frames before the shot's end, drawing that ending itself. |

The 22-frame options exist because of roycho87. After seeing this repo he pointed out that 5 frames is not enough for H3 to get a good idea of the motion, so it can easily change the speed, and that the last frames of a clip tend to be rigid, which is exactly what a bridge here was starting from. His own settings (Geiru Context 39 / `drop_last` 17 and Geiru Anchoring 39 / `drop_first` 17) are the third option. He was right: on the running test, with the same shots and only the bridges re-rendered, the pace change between pieces went away.

After using both, this is where I've landed:

- **5 frames** still works well for transitions in a storyboard: short filler actions without sustained movement, or a gap between two actions. With 22 you would cut a good part of a short clip off.
- **22 frames** is for long-form, continuous movement, where the same pace has to carry through a shot, the bridge and the next shot. It gives much better results, more reliably, and it fixed the slowdown.
- If you don't mind rendering the extra frames and timing around them, 22 will probably always perform better. It just isn't always needed.

One bridge can go its own way with a word on its storyboard line. This video uses 5 everywhere except the stretch where she keeps walking:

```
> She picks up the cup.
~ She sets it down and turns to the door.
> She walks down the hall.
~ hold 22 remake She keeps walking, past the windows.
> She reaches the far door.
```

A hold only belongs to its bridge, so changing it re-renders that one bridge and nothing else. Both versions stay in the cache, so you can flip back and forth to compare, and `solo` on the same line shows you just that transition.

What it costs. Lengths are what H3 renders, so held frames come out of what a piece adds to the video:

| Piece set to | `5 frames` | `22 frames, re-make the shot's last 17` |
|---|---|---|
| 5 s shot | all 5 s is the shot's own | 4.5 s shown (its last 17 frames are the bridge's version) |
| 5 s bridge | 4.75 s of its own | 3.3 s of its own |
| 10 s shot | 10 s | 9.4 s |
| 10 s bridge | 9.7 s | 8.3 s |

With 22, a shot or bridge has to be about 2.3 seconds or longer, and about 3 seconds with the re-make. The node stops with a message naming the piece if one is too short.

## Other settings

| Setting | What it does |
|---|---|
| `shots` | How many shots, when the storyboard has no pieces. |
| `shot_seconds`, `bridge_seconds` | Lengths for pieces that don't give their own. |
| `soundtrack` | `generated` (H3's audio, crossfaded at the joins) or `silent`. |
| `upscale_pass` | Latent upscale and a short polish per piece. Needs the Latent Upscaler pack. |
| `frame_labels` | Burns the piece name and frame number into every frame, for review. |
| `bridge_hold` | How much of its neighbours a bridge holds. See "How much a bridge holds". |
| `sampler`, `sigmas` | Optional inputs. Not wired, the node uses er_sde, beta, 6 steps (set up for the 4-step turbo LoRA). |
| `ref_video`, `ref_audio` | Optional references, `<Video 1>` / `<Audio 1>`. A short reference video is looped along the whole timeline. |

The **plan** output lists every piece with its frames, its pictures and any warnings. Long runs are written to `output/h3_rolling/rolling_<time>.mp4` and the node's `frames` output becomes a 240p proxy.

## What I learned using it

- **The big one: where a bridge lands in the next shot.** This was the main thing that broke. With a subject in continuous motion (a run), about a third of the joins jumped at the point where a bridge reached the next shot. It never happened where a bridge left a shot. My first fix was regenerating the bridge on new seeds until one landed well, which sort of worked but was a lottery. The real fix was pinning the bridge further into the next shot: 17 frames in, not on its first frame. After that the jumps were gone with no re-rolling, and the node now always does it. The clips above show the same join both ways, and the measurement behind it is under "What might be worth a look".
- **Longer pieces tend to be paced better.** As a rule of thumb, the longer the clip, the more likely H3 paces the movement correctly. That is how H3 times things, not something this node does. Very short pieces spend most of their length getting into and out of their neighbours.
- **Size each piece by what happens in it.** Two big chalk words took about 11 seconds. Asked to do it in less, the letters appeared faster than the hand moved.
- **Put the action that changes the scene in a bridge**, and the small actions around it (turning, picking something up) in the shots on either side.
- **A place picture carries the room, not the camera.** Without a framing sentence that every piece shares, four shots gave four camera positions.
- **Say "static, locked-off"** if you don't want a long bridge to push in and pull back out.
- **A bad first frame can be the seed, not the words.** Re-roll that one shot with `solo`.
- **Dialogue inside a shot works** with `<d>[English] ...</d>` in the piece's text. I have not tested a line that crosses a join.

## Where it's weak

- **Pace between pieces, with the default 5-frame holds.** A walk or a run can change speed from one piece to the next. Holding 22 frames fixed this in my tests (see "How much a bridge holds"). What can still happen is one shot picking an odd tempo on its own, since every shot is a fresh render. Re-roll that shot (`solo`, `take 2`), or give every piece a looped motion reference video.
- **Two of Roy's rules that this node doesn't follow, untested here.** His README recommends clip lengths that are multiples of 3, because held audio otherwise builds up down a chain. Nothing is chained here, so it may not matter, but I haven't checked. His notes also say to hold more frames at the end of a clip than at its start. This node holds the same number at both ends.

---

# H3 Rolling Ref2VA

Long video-to-video in one node. H3 is trained on 5-15 second clips. The usual way to go longer is to chain continuations, and over many hand-offs contrast, sheen and blemishes compound. This node never chains: every chunk is rendered fresh against its own slice of the source video, and each seam gets a short bridge clip pinned to the end of one chunk and the start of the next.

- **Works well:** one subject who stays in frame, with a fairly steady camera.
- **Still rough:** the subject leaving or entering frame, crowds and several similar-looking people, and big fast moves that land right on a seam.

On a 70 second test clip with structure lock on, 8 of 10 seams were close to invisible and the two that weren't were head turns.

### H3 Rolling Source (scrub)
Pick or upload a video, scrub the built-in player and set the start and end points. It outputs frames at 24 fps (any input frame rate is resampled) and the soundtrack. `short_edge` shrinks the source: it's only choreography, and 480 is plenty for most clips.

**Queue batch** (button under the player): splits the selected range into slices of the Rolling Ref2VA node's `chunk_seconds` and queues one single-chunk job per slice. No bridges, no stitching.

### H3 Rolling Ref2VA

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

The prompt goes to every chunk and uses the normal H3 tags: `<Picture N>` for your pictures, `<Video 1>` for the source slice, `<Audio 1>` for its soundtrack. The default prompt is a worked example.

| Setting | What it does |
|---|---|
| `megapixels` | Output size. Aspect comes from the source. 0.4 = 864x480 at 16:9. |
| `chunk_seconds` | Length of each independent chunk. 5 is a good default. |
| `bridge_frames` | Length of each seam bridge. |
| `pin_mode` | How a bridge holds the frames it shares with its chunks. `keyframe`: passed as H3 keyframe guides, joins are short crossfades. `mask`: the chunks' real latents held under a mask, joins are plain cuts. |
| `soundtrack` | `generated`, `source` (the original audio untouched) or `silent`. |
| `handoff` | `bridge` is the default. `pixel` and `latent` chain each chunk off the previous one and `none` crossfades independent chunks; they are there for comparison. |

### Structure lock
Independent chunks can disagree on framing. With structure lock on, SAM3 tracks the person being replaced, the node draws an edge map of the source with that person cut out, and feeds it to the H3 Fun ControlNet as a control video. The background edges pin camera, framing and zoom to the source while the person is drawn fresh.

- Needs the SAM3 checkpoint in `models/checkpoints` (`sam3.1_multiplex_fp16.safetensors`) and the H3 Fun ControlNet Union patch in `models/model_patches` (`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors`).
- `lock_track`: who to track, in plain words. `lock_grow`: slack around them. `lock_max_people`: how many matches SAM3 may keep.
- `lock_preview_only` writes a preview of the tracking and stops. Worth checking before a long render.

### Per-chunk dialogue
With `dialogue` on, the node transcribes the source soundtrack (WhisperX, word timestamps, speaker diarization) and gives every chunk and bridge only the words spoken inside its own frames, as `<d>[English] ...</d>`.

- `dialogue = preview only` transcribes, prints every piece's lines and renders nothing.
- `speakers` says who each label is, one per line: `SPEAKER_00 = <Subject 1>`.
- Needs WhisperX, either in ComfyUI's Python or in another Python named by the `H3_WHISPER_PY` environment variable. Diarization needs `HF_TOKEN`.

### Prompting tips
- Don't describe camera moves in the prompt. The same prompt goes to every chunk, so "slow push-in" makes every chunk zoom.
- For character swaps, describe what the new character looks like and leave the motion to `<Video 1>`.

---

## Credits and related work

- **chanon**, [comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline): the timeline editor with extend, prepend, bridge and loop from saved latents. Its docs are the best written account I've found of how these joins behave.
- **roycho87**, [Geiru Context Suite](https://github.com/roycho87/geiru_context_suite) and [NerdNotes](https://github.com/roycho87/NerdNotes): Context, Anchoring, Keystone, the 17-frame block rules, and a lot of careful measurement. The masked hold in this pack follows his nodes' method, and the 22-frame holds come from his diagnosis of this node's pacing.
- **drozbay** and **seitanism** for the masked-latent method itself, and **kijai** for ComfyUI's H3 support.
- The [Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) used by `upscale_pass`.

If something here is wrong about your work, or I've missed crediting someone, open an issue and I'll fix it.

## License

MIT
