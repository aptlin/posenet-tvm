import torch
from posenet.utils import argmax2d
from posenet.constants import NUM_KEYPOINTS

# import * as tf from '@tensorflow/tfjs-core';

# import {partNames} from '../keypoints';
# import {Keypoint, Pose, PoseNetOutputStride} from '../types';

# import {argmax2d} from './argmax2d';
# import {getOffsetPoints, getPointsConfidence} from './util';

# /**
#  * Detects a single pose and finds its parts from part scores and offset
#  * vectors. It returns a single pose detection. It works as follows:
#  * argmax2d is done on the scores to get the y and x index in the heatmap
#  * with the highest score for each part, which is essentially where the
#  * part is most likely to exist. This produces a tensor of size 17x2, with
#  * each row being the y and x index in the heatmap for each keypoint.
#  * The offset vector for each for each part is retrieved by getting the
#  * y and x from the offsets corresponding to the y and x index in the
#  * heatmap for that part. This produces a tensor of size 17x2, with each
#  * row being the offset vector for the corresponding keypoint.
#  * To get the keypoint, each part’s heatmap y and x are multiplied
#  * by the output stride then added to their corresponding offset vector,
#  * which is in the same scale as the original image.
#  *
#  * @param heatmapScores 3-D tensor with shape `[height, width, numParts]`.
#  * The value of heatmapScores[y, x, k]` is the score of placing the `k`-th
#  * object part at position `(y, x)`.
#  *
#  * @param offsets 3-D tensor with shape `[height, width, numParts * 2]`.
#  * The value of [offsets[y, x, k], offsets[y, x, k + numParts]]` is the
#  * short range offset vector of the `k`-th  object part at heatmap
#  * position `(y, x)`.
#  *
#  * @param outputStride The output stride that was used when feed-forwarding
#  * through the PoseNet model.  Must be 32, 16, or 8.
#  *
#  * @return A promise that resolves with single pose with a confidence score,
#  * which contains an array of keypoints indexed by part id, each with a score
#  * and position.
#  */


def decode_single_pose(scores, offsets, output_stride):
    total_score = 0.0

    heatmap_max_score_coords = argmax2d(scores)

    # change dimensions from (2k, h, w) to (k, h, w, 2)
    offsets = (
        offsets.reshape(2, -1, offsets.shape[1], offsets.shape[2])
        .transpose(0, 1)
        .transpose(1, 2)
        .transpose(2, 3)
    )

    # get offset vectors
    offset_vectors = offsets[
        torch.arange(NUM_KEYPOINTS),
        heatmap_max_score_coords[:, 0],
        heatmap_max_score_coords[:, 1],
        :,
    ]
    keypoints = (heatmap_max_score_coords * output_stride).float() + offset_vectors

    keypoint_confidence = scores[
        torch.arange(NUM_KEYPOINTS),
        heatmap_max_score_coords[:, 0],
        heatmap_max_score_coords[:, 1],
    ]

    confidence = keypoint_confidence.mean()

    return (
        keypoints.to("cpu").numpy(),
        confidence.to("cpu").numpy(),
        keypoint_confidence.to("cpu").numpy(),
    )
