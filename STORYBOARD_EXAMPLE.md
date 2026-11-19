# A full storyboard, with every control in use

This is one scene written out for the **H3 Rolling Timeline** node, using every setting the storyboard box has, with the reason for each choice. It is meant to be read top to bottom by a person, or handed whole to an agent that is going to write storyboards.

The two boxes are also in this repo as plain files you can paste: [`examples/classroom_prompt.txt`](examples/classroom_prompt.txt) and [`examples/classroom_storyboard.txt`](examples/classroom_storyboard.txt).

**What has and hasn't been rendered.** The first five pieces (the chalkboard writing) are the storyboard behind the chalkboard clip in the README, with one sentence of the shared prompt shortened. The other fourteen pieces were written to show the controls working together. The whole file has been run through the node's parser and length checks, so it is valid as typed, but pieces 6 to 19 have not been rendered. Treat their wording as a starting point, not as tested takes.

## The scene

A woman at the front of a lecture room. She says a line, writes two lines on the chalkboard, points at them, paces while she thinks, wipes the board clean, asks for questions and takes a bow. About two minutes, one fixed camera.

## How to think about it before typing anything

1. **Write the beats first, in order.** One action or one small group of actions per beat. Don't think about shots and bridges yet.
2. **Alternate them: shot, bridge, shot.** Every piece is a real clip with its own action, so nothing is "just a transition". But the two kinds behave differently:
   - A **shot** is rendered from the pictures and its words, knowing nothing about its neighbours.
   - A **bridge** is rendered afterwards, locked at its start to the shot before it and at its end to the shot after it.
3. **Put anything that changes the scene in a bridge.** A bridge knows where it has to end up. If the board has to go from blank to written, the bridge is the piece that can get there. The small actions around it (turning, picking up the chalk, putting it down) go in the shots on either side.
4. **Size each piece by what happens in it.** Mime it. Writing two big words takes about eleven seconds; a nod takes two. A piece that is too short for its action rushes it, and one that is too long invents filler.
5. **Decide how each bridge grips its neighbours.** Short action with a clear start and stop: hold 5 frames. Movement that has to keep the same pace from one piece into the next: hold 22.
6. **Decide which pictures each stretch needs.** Pictures carry forward until you change them, like props left on a set.

## The pictures

Four inputs on the node:

| Input | What it is |
|---|---|
| `picture_1` | The character: a multi-view sheet of one woman. |
| `picture_2` | The room, chalkboard blank. |
| `picture_3` | The same room, first line written on the board. |
| `picture_4` | The same room, both lines written. |

Pictures 2, 3 and 4 are the same image apart from the chalk. That matters: the room must not shift when the picture changes.

## The node's settings

| Setting | Value | Why |
|---|---|---|
| `shot_seconds` | 5 | Used by any shot that doesn't give a length. |
| `bridge_seconds` | 5 | Same, for bridges. |
| `bridge_hold` | `5 frames` | Most joins here are short storyboard transitions. The pacing stretch overrides it per bridge. |
| `seed` | 4242 | Shot N gets seed + N - 1 unless it has `take` or `seed`. |
| `upscale_pass` | off | Work at preview size; turn it on at the end. Finished pieces are reused. |
| `frame_labels` | on while working | Lets you say "shot 4, frame 60". |

## The prompt box

Everything that is true of every piece goes here, once. `{beat}` is where each piece's own text lands.

```
subject_definitions:

summary:
[reference generation] {beat}

retention_analysis:

detailed_description:
Photorealistic live-action footage, filmed by a static camera.

[Shot 1] A static, locked-off, straight-on view of the front wall of <Subject 2>, showing the chalkboard, the lectern and the projector screen. {beat}

overall_soundscape:
Quiet room tone.

non_diegetic_music:
N/A
```

Things to notice:

- `subject_definitions:` and `retention_analysis:` are empty. The node fills them in for each piece from the `image` lines below.
- The camera is described here and nowhere else. A picture of a room gives H3 the room, not the camera position, so without this sentence every shot picks its own framing. "Static, locked-off" stops a long bridge from pushing in and pulling back.
- The framing sentence says nothing about where she is, because she moves. Each beat says where she stands.
- It does not describe the room or the woman. The pictures do that.

## The storyboard box

```
image 1: <Subject 1> is the clown woman from the multi-view sheet <Picture>. The sheet shows several views of one woman; there is one woman.
  keep: her appearance and clothing from <Picture>
image 2, 3, 4: <Subject 2> is the lecture room in <Picture>.
  keep: the room in <Picture>
start: image 1, image 2

> 6.5s seed 5001 <Subject 1> (S1) stands facing the camera with both hands on the top of the lectern and says, <d>[English] Blackboard test, take one.</d> She then turns around to face the clean, empty chalkboard, picks up a stick of chalk from the tray and holds it up near the top left of the chalkboard.

~ 11s <Subject 1> stands facing the chalkboard with her back to the camera and writes "LATENT SPACE" with the stick of chalk in large capital letters across the upper half of the chalkboard.

> 1.5s image 3 -image 2 <Subject 1> stands facing the chalkboard with her back to the camera, lifts the stick of chalk off the chalkboard and lowers her hand to the left side of the chalkboard, below the line of writing, holding the chalk just off the surface.

~ 277f <Subject 1> stands facing the chalkboard with her back to the camera and writes "IS A TRAP!" with the stick of chalk in large capital letters across the chalkboard, below the line already written.

> image 4 -image 3 <Subject 1> puts the stick of chalk back in the tray, turns around to face the camera and rests both hands on the top of the lectern.

~ <Subject 1> steps out from behind the lectern and walks to the left end of the chalkboard.

> 7s <Subject 1> (S1) stands at the left end of the chalkboard, points at the writing with her right hand and says, <d>[English] Nobody warned me. So I am warning you.</d> She lowers her hand.

~ 10s hold 22 remake <Subject 1> paces slowly along the front of the chalkboard, from its left end to its right end and back.

> 10s take 2 <Subject 1> paces slowly back and forth in front of the chalkboard.

~ 10s hold 22 remake <Subject 1> paces slowly back and forth in front of the chalkboard.

> 10s <Subject 1> paces slowly back and forth in front of the chalkboard, then stops in front of the writing and holds still for a beat.

~ 9s <Subject 1> picks up the eraser from the chalk tray and wipes the writing off the chalkboard with broad strokes, leaving the chalkboard clean.

> 4s image 2 -image 4 <Subject 1> puts the eraser back in the chalk tray and claps the chalk dust from her hands.

~ hold 22 <Subject 1> walks back to the lectern and stands behind it.

> 6s <Subject 1> (S1) stands behind the lectern with both hands on its top, looks at the camera and says, <d>[English] Any questions?</d> She holds still for a beat.

~ 3s <Subject 1> straightens up and steps out to stand beside the lectern.

> 3s <Subject 1> stands beside the lectern and takes a small bow.

~ 3s <Subject 1> straightens up from the bow.

> <Subject 1> stands beside the lectern, smiles at the camera and holds still for a beat.
```

## How the syntax works

**The top block describes the pictures.** It is optional; without it, pictures are simply handed over and you write `<Picture 1>` yourself in the prompt.

- `image 1: ...` is the definition line for `picture_1`. Write `<Picture>` with no number where the sentence points at the picture. The node fills in the number for each piece.
- `keep: ...` under it is what H3 should preserve from that picture. Left off, it becomes "as shown in `<Picture N>`".
- `image 2, 3, 4: ...` gives three pictures the same description. They are three states of one room, so they are all `<Subject 2>`.
- `start: image 1, image 2` is what the first piece is handed. Left off, every loaded picture is handed over.

**Each piece starts with a marker.** `>` is a shot, `~` is a bridge. They must alternate and the storyboard must begin and end on a shot; the node stops with a message if two of the same kind are in a row, which is how a missing line gets caught.

**Settings come right after the marker, as bare lowercase words, before the text:**

| Word | Meaning | Carries forward? |
|---|---|---|
| `7s` | Length in seconds, snapped to the nearest length H3 can make (17k + 5 frames). | No. This piece only. |
| `277f` | Length in frames, for when you want an exact one. | No. |
| `image 3` | Start handing `picture_3` over. | **Yes**, until removed. |
| `-image 2` | Stop handing `picture_2` over. | **Yes.** |
| `take 2` | A different roll of this piece. | No. |
| `seed 5001` | This exact seed for this piece. | No. |
| `hold 5` / `hold 22` / `hold 22 remake` | Bridges only: how this bridge grips its two shots. | No. |
| `solo` | Render only this piece. | No. |

The rule behind the last column: pictures are part of the scene, so they stay until something changes them. Everything else is a decision about one piece.

**Everything after the settings, up to the next marker, is that piece's text.** Line breaks don't matter. The blank lines between pieces above are only for reading.

## Piece by piece, and why

| # | Piece | Settings | Why |
|---|---|---|---|
| 1 | Shot 1: speaks, turns, picks up chalk | `6.5s seed 5001` | Timed to a line of dialogue plus two small actions. `seed 5001` pins a roll that was kept after trying a few: the default seed left a smudge on the board in the first frame. |
| 2 | Bridge 1: writes the first line | `11s` | The change to the room happens here. This bridge still has the blank-board picture, but it ends locked to shot 2, which was rendered from the picture with the line written. So she writes toward it. |
| 3 | Shot 2: moves her hand to the second line | `1.5s image 3 -image 2` | A very short shot between two bridges, so one long action (writing two lines) is split in two without a pause. From here on the room is the one-line picture. The text says exactly what the hand does; "moves the chalk down" drew an underline. |
| 4 | Bridge 2: writes the second line | `277f` | 11.5 seconds, given in frames to show the other form. It inherits the one-line picture from shot 2 and ends locked to shot 3. |
| 5 | Shot 3: puts the chalk down, turns | `image 4 -image 3` | No length, so it uses `shot_seconds`. The room is now the two-line picture. |
| 6 | Bridge 3: walks to the board | (none) | Default length, default 5-frame hold. A short move with a clear start and end. |
| 7 | Shot 4: points and speaks | `7s` | Dialogue goes inside a shot, with the speaker tag `(S1)` on the same speaker each time. |
| 8 | Bridge 4: starts pacing | `10s hold 22 remake` | From here the movement is continuous, so the bridge holds 22 frames of each neighbour and re-draws the last 17 of shot 4. Ten seconds, because long continuous movement paces better in longer pieces. |
| 9 | Shot 5: pacing | `10s take 2` | `take 2` stands for "the first roll of this shot came out slow, so I rolled again and kept the second". |
| 10 | Bridge 5: pacing | `10s hold 22 remake` | Same pace has to carry from shot 5 into shot 6. |
| 11 | Shot 6: pacing, then stops | `10s` | Ends on a held beat, so the next piece starts from stillness. |
| 12 | Bridge 6: wipes the board | `9s` | The room changes back. The bridge keeps the two-line picture and ends locked to a shot rendered from the blank one. Back to the default hold: she starts from a standstill. |
| 13 | Shot 7: puts the eraser down | `4s image 2 -image 4` | Pictures can go back to an earlier one. |
| 14 | Bridge 7: walks to the lectern | `hold 22` | A walk, so 22 frames, but without `remake`: shot 7 is short and its ending is worth keeping. |
| 15 | Shot 8: asks for questions | `6s` | |
| 16 | Bridge 8: steps out | `3s` | Three quick 3 second pieces follow. This is what the 5-frame hold is for: with 22, most of a 3 second piece would be copies of its neighbours. |
| 17 | Shot 9: bows | `3s` | She is beside the lectern, not behind it. A bow behind a lectern goes into the lectern. |
| 18 | Bridge 9: straightens | `3s` | |
| 19 | Shot 10: smiles, holds | (none) | Default length. A held beat gives the video an ending. |

Lengths as the node snaps them: 6.6, 10.8, 1.6, 11.5, 5.2, 5.2, 7.3, 10.1, 10.1, 10.1, 10.1, 8.7, 3.8, 5.2, 5.9, 3.0, 3.0, 3.0 and 5.2 seconds. The finished video is about 1 minute 57 seconds, because each bridge overlaps its two shots by the frames it holds.

## What H3 actually receives

The node builds one prompt per piece. This is piece 13 (shot 7), which is handed `picture_1` and `picture_2`:

```
subject_definitions:
<Subject 1> is the clown woman from the multi-view sheet <Picture 1>. The sheet shows several views of one woman; there is one woman.
<Subject 2> is the lecture room in <Picture 2>.

summary:
[reference generation] <Subject 1> puts the eraser back in the chalk tray and claps the chalk dust from her hands.

retention_analysis:
<Subject 1> (throughout [Shot 1]): fully_preserved - her appearance and clothing from <Picture 1>.
<Subject 2> (throughout [Shot 1]): fully_preserved - the room in <Picture 2>.

detailed_description:
Photorealistic live-action footage, filmed by a static camera.

[Shot 1] A static, locked-off, straight-on view of the front wall of <Subject 2>, showing the chalkboard, the lectern and the projector screen. <Subject 1> puts the eraser back in the chalk tray and claps the chalk dust from her hands.

overall_soundscape:
Quiet room tone.

non_diegetic_music:
N/A
```

Two things happened that you didn't type:

- The definition and retention lines were written from the `image` lines, only for the pictures this piece has.
- The picture numbers were filled in by position. H3 numbers pictures in the order it receives them. In piece 5, which has `picture_1` and `picture_4`, the room is also `<Picture 2>`, because it is the second picture that piece was given.

Nothing about pictures a piece doesn't have appears in its prompt.

## Working on it

You would not render all nineteen pieces and hope. The order that wastes the least time:

1. **Trial the first shot alone.** Add `solo` to its line. Only that piece renders.

   ```
   > 6.5s solo <Subject 1> (S1) stands facing the camera ...
   ```

   If the roll is bad, add `take 2`, then `take 3`. When one is good, remove `solo` and leave the `take`. The full run will find that piece in the cache and not render it again.

2. **Run the whole thing at preview size.** Look at it with `frame_labels` on.

3. **Fix pieces one at a time.** Change a shot's text, length, pictures or take, and run again. That shot and the two bridges touching it render; everything else comes back from the cache. The last line of the **plan** output tells you what was rendered and what was reused.

4. **Compare holds on one join.** Put `solo` on a bridge and it renders with its two shots (which come from the cache), so you see just that transition:

   ```
   ~ 10s solo hold 22 remake <Subject 1> paces slowly ...
   ```

   Change `hold 22 remake` to `hold 5` and run again. Only the bridge re-renders, and both versions stay cached, so switching back costs nothing.

5. **Turn `upscale_pass` on last.** Pieces already rendered at preview size are upscaled from the cache, so you get the same performance you approved, larger.

## Things that will stop the run, and things that won't

Stops with a message naming the piece:

- Two shots or two bridges in a row, or a storyboard that ends on a bridge.
- A piece too short for its holds. With `hold 22` a shot or bridge needs about 2.3 seconds; with `hold 22 remake`, about 3.

Carries on, with a `note:` line in the plan:

- A piece mentions a picture it isn't being handed.
- `hold` on a shot line (it only means something on a bridge).
- `image 5` when nothing is plugged into `picture_5`.

Does nothing at all, by design:

- Leaving out the `image` lines, `keep`, `start`, lengths, or the whole storyboard. Nothing is added to the prompt for parts you didn't type.

## Not shown here

- **`ref_video` and `ref_audio`.** If you plug either in, every piece gets it as `<Video 1>` / `<Audio 1>`, and a short reference video is looped along the whole timeline. Use `<Video 1>` in the prompt the way you would in any H3 prompt.
- **Running without a storyboard.** Leave the storyboard box empty, put a prompt with no `{beat}` in the prompt box, and set `shots`. Every piece gets the same prompt.
- **A line of dialogue that crosses from one piece into the next.** Untested. Keep each spoken line inside one shot.
