import argparse
import os
from timeit import default_timer as now

import cv2
import numpy as np
import torch
from tqdm import tqdm

import posenet
from utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=101)
parser.add_argument("--decoder", type=str, default="multi")
parser.add_argument("--scale_factor", type=float, default=1.0)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--image_dir", type=str, default="./images")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--force-cpu", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--use-tvm", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--resize", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--input-name", type=str, default="image")
parser.add_argument("--processing-width", type=int, default=600)
parser.add_argument("--processing-height", type=int, default=340)
parser.add_argument(
    "--tvm-graph", type=str, default="./build/deploy_graph_image_600_340.json"
)
parser.add_argument(
    "--tvm-lib", type=str, default="./build/deploy_lib_image_600_340.tar"
)
parser.add_argument(
    "--tvm-params", type=str, default="./build/deploy_params_image_600_340.params"
)
args = parser.parse_args()

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.force_cpu
    else torch.device("cpu")
)


def main():
    model = posenet.load_model(args.model)
    model = model.to(DEVICE).eval()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path
        for f in os.scandir(args.image_dir)
        if f.is_file() and f.path.endswith((".png", ".jpg"))
    ]

    if args.use_tvm:
        import tvm
        from tvm.contrib import graph_runtime

        with open(args.tvm_graph) as f:
            tvm_graph = f.read()
        tvm_lib = tvm.runtime.load_module(args.tvm_lib)
        with open(args.tvm_params, "rb") as f:
            tvm_params = bytearray(f.read())
        ctx = tvm.cpu()
        module = graph_runtime.create(tvm_graph, tvm_lib, ctx)
        module.load_params(tvm_params)

    preprocessing_time = []
    inference_time = []
    processing_time = []

    for f in tqdm(filenames, desc="Processed", unit="files"):
        start = now()
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f,
            scale_factor=args.scale_factor,
            output_stride=output_stride,
            resize=(args.processing_width, args.processing_height)
            if args.resize
            else None,
        )
        preprocessing_time.append(now() - start)

        start = now()
        with torch.no_grad():
            if args.use_tvm:
                input_data = tvm.nd.array(input_image)
                module.run(**{args.input_name: input_data})
                out = []
                for idx in range(module.get_num_outputs()):
                    res = (
                        torch.Tensor(module.get_output(idx).asnumpy())
                        .squeeze(0)
                        .to(DEVICE)
                    )
                    out.append(res)

            else:
                input_image = torch.Tensor(input_image).to(DEVICE)

                out = []
                for idx, res in enumerate(model(input_image)):
                    out.append(res.squeeze(0))

            inference_time.append(now() - start)

            (
                heatmaps_result,
                offsets_result,
                displacement_fwd_result,
                displacement_bwd_result,
            ) = out

            start = now()
            if args.decoder == "multi":
                (
                    pose_scores,
                    keypoint_scores,
                    keypoint_coords,
                ) = posenet.decode_multiple_poses(
                    heatmaps_result,
                    offsets_result,
                    displacement_fwd_result,
                    displacement_bwd_result,
                    output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.25,
                )
            elif args.decoder == "single":
                (keypoints, pose_score, keypoint_scores) = posenet.decode_single_pose(
                    heatmaps_result, offsets_result, output_stride
                )
                pose_scores = np.asarray([pose_score])
                keypoint_scores = np.asarray([keypoint_scores])
                keypoint_coords = np.asarray([keypoints])

            else:
                raise NotImplementedError(
                    "The decoder {} is not implemented.".format(args.decoder)
                )
            processing_time.append(now() - start)

        keypoint_coords *= output_scale

        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image,
                pose_scores,
                keypoint_scores,
                keypoint_coords,
                min_pose_score=0.25,
                min_part_score=0.25,
            )

            cv2.imwrite(
                os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)),
                draw_image,
            )

        if args.verbose:
            print("Results for image: %s" % f)
            for point_idx in range(len(pose_scores)):
                if pose_scores[point_idx] == 0.0:
                    break
                print("Pose #%d, score = %f" % (point_idx, pose_scores[point_idx]))
                for keypoint_idx, (score, coord) in enumerate(
                    zip(keypoint_scores[point_idx, :], keypoint_coords[point_idx, :, :])
                ):
                    print(
                        "Keypoint %s, score = %f, coord = %s"
                        % (posenet.PART_NAMES[keypoint_idx], score, coord)
                    )

    avg_preprocessing_time = np.mean(preprocessing_time)
    avg_postprocessing_time = np.mean(processing_time)
    avg_inference_time = np.mean(inference_time)
    print("=" * 80)
    print(
        "Decoder: {}, TVM Runtime: {}, Resize to {}x{} HxW: {}".format(
            args.decoder,
            "enabled" if args.use_tvm else "disabled",
            args.processing_height,
            args.processing_width,
            "enabled" if args.resize else "disabled",
        )
    )
    print("-" * 80)

    print("Average pre-processing FPS: {:.2f}".format(1 / avg_preprocessing_time))
    print("Average inference FPS: {:.2f}".format(1 / avg_inference_time))
    print("Average post-processing FPS: {:.2f}".format(1 / avg_postprocessing_time))
    print(
        "Average FPS: {:.2f}".format(
            1 / (avg_postprocessing_time + avg_inference_time + avg_preprocessing_time)
        )
    )


if __name__ == "__main__":
    main()
