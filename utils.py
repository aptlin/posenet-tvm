import cv2
from tqdm import tqdm
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def frame_iter(capture, description):
    def _iterator():
        while capture.grab():
            yield capture.retrieve()[1]

    return tqdm(
        _iterator(),
        desc=description,
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        unit="frames",
    )


def make_video(
    output_filename, images, fps=30, size=None, is_color=True, format_code="FMP4"
):
    fourcc = cv2.VideoWriter_fourcc(*format_code)
    vid = None
    for image in images:
        if vid is None:
            if size is None:
                size = image.shape[1], image.shape[0]
            vid = cv2.VideoWriter(output_filename, fourcc, float(fps), size, is_color)
        if size[0] != image.shape[1] and size[1] != image.shape[0]:
            img = cv2.resize(image, size)
        vid.write(image)
    vid.release()
    return vid
