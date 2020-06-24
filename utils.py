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
