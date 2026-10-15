"""Extract left-camera MP4s from teleop demo SVO recordings.

openpi's DROID -> LeRobot converter reads recordings/MP4/<serial>.mp4 (left view of each
stereo camera, BGR, one frame per trajectory step). Reading SVOs needs the ZED SDK, so run
this on the robot laptop:

    ZED_SDK_SVO_VERSION=1 python finetune/svo_to_mp4.py ~/eric/demos

Only kept demos (demo_<n>/) are converted; partial and in-progress recordings are skipped.
MP4s are written to a temp file and only moved into place once complete, so existing MP4s
are skipped and it is safe to re-run.
"""
import glob
import os
import re
import sys

import cv2
import pyzed.sl as sl

DROID_FPS = 15


def mp4_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    cap.release()
    return count


def svo_to_mp4(svo_path: str, mp4_path: str) -> int:
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE

    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open {svo_path}: {err}")

    resolution = cam.get_camera_information().camera_configuration.resolution
    tmp_path = mp4_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), DROID_FPS, (resolution.width, resolution.height))

    image = sl.Mat()
    frames = 0
    while cam.grab() == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(image, sl.VIEW.LEFT)
        writer.write(image.get_data()[:, :, :3])  # BGRA -> BGR
        frames += 1

    writer.release()
    cam.close()

    if mp4_frame_count(tmp_path) != frames:
        raise RuntimeError(f"Frame count mismatch writing {mp4_path}")
    os.replace(tmp_path, mp4_path)
    return frames


def main(root: str) -> None:
    svos = sorted(glob.glob(os.path.join(os.path.expanduser(root), "**", "recordings", "SVO", "*.svo"), recursive=True))
    svos = [s for s in svos if re.fullmatch(r"demo_\d+", s.split(os.sep)[-4])]
    print(f"Found {len(svos)} SVO recordings in kept demos")

    for svo in svos:
        demo_dir = os.path.dirname(os.path.dirname(os.path.dirname(svo)))
        mp4_dir = os.path.join(demo_dir, "recordings", "MP4")
        os.makedirs(mp4_dir, exist_ok=True)
        mp4 = os.path.join(mp4_dir, os.path.basename(svo)[: -len(".svo")] + ".mp4")
        name = os.path.relpath(mp4, os.path.expanduser(root))

        if os.path.exists(mp4):
            print(f"  skip (exists, {mp4_frame_count(mp4)} frames): {name}")
            continue

        frames = svo_to_mp4(svo, mp4)
        print(f"  {frames} frames -> {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/eric/demos")
