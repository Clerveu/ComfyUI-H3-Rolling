"""Small copies of Royale takes for the results page's grid: 270 px tall, a keyframe every half second
(so many of them play at once and scrub quickly). The page uses <take>.proxy.mp4 when it exists and the
full file in the A/B view.

As a script: python royale_proxy.py <run folder or the h3_royale folder>   makes any that are missing.
"""
import os
import shutil
import subprocess
import sys


def proxy_path(path):
    return path[:-4] + ".proxy.mp4"


def make_proxy(path):
    """Returns the proxy's path, or None if ffmpeg isn't there or failed."""
    exe = shutil.which("ffmpeg")
    if exe is None or not os.path.isfile(path):
        return None
    out = proxy_path(path)
    tmp = out[:-4] + ".tmp.mp4"
    r = subprocess.run([exe, "-v", "error", "-y", "-i", path, "-vf", "scale=-2:270", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-g", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", tmp])
    if r.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    made = 0
    for root, _, files in os.walk(sys.argv[1]):
        for f in sorted(files):
            if f.startswith("arm") and f.endswith(".mp4") and ".proxy." not in f and ".tmp." not in f:
                p = os.path.join(root, f)
                if not os.path.exists(proxy_path(p)) and make_proxy(p):
                    made += 1
                    print("made", proxy_path(p), flush=True)
    print(made, "proxies made")
