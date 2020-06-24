import argparse
import os
from timeit import default_timer as now

import cv2
import numpy as np
import torch
from tqdm import tqdm

import posenet
from utils import str2bool, frame_iter

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=101)
parser.add_argument("--decoder", type=str, default="multi")
parser.add_argument("--scale_factor", type=float, default=1.0)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--videos_dir", type=str, default="./videos")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--force-cpu", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--use-tvm", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--webcam", type=str2bool, nargs="?", const=True, default=False)
parser.add_argument("--cam_id", type=int, default=0)
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
parser.add_argument("--output-format", type=str, default="FMP4")
args = parser.parse_args()

DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.force_cpu
    else torch.device("cpu")
)


def process_capture(model, capture, **kwargs):
    output_stride = model.output_stride

    preprocessing_time = []
    inference_time = []
    processing_time = []

    decoded_images = []
    for frame in frame_iter(capture, "Progress"):
        if args.webcam:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        start = now()
        input_image, draw_image, output_scale = posenet.process_input(
            frame,
            scale_factor=args.scale_factor,
            output_stride=output_stride,
            resize=(args.processing_height, args.processing_width)
            if args.resize
            else None,
        )
        preprocessing_time.append(now() - start)

        start = now()
        with torch.no_grad():
            if args.use_tvm:
                import tvm

                module = kwargs.get("module", None)
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
            if args.webcam:
                cv2.imshow("frame", draw_image)
            else:
                decoded_images.append(draw_image)
    return decoded_images, preprocessing_time, inference_time, processing_time


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


def main():
    model = posenet.load_model(args.model)
    model = model.to(DEVICE).eval()

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

    if not args.webcam:
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path
            for f in os.scandir(args.videos_dir)
            if f.is_file() and f.path.endswith((".mp4"))
        ]

        preprocessing_time = []
        inference_time = []
        processing_time = []

        for filename in tqdm(filenames, desc="Processed", unit="files"):
            cap = cv2.VideoCapture(filename)
            if args.use_tvm:
                out = process_capture(model, cap, module=module)
            else:
                out = process_capture(model, cap)

            (
                decoded_images,
                video_preprocessing_time,
                video_inference_time,
                video_processing_time,
            ) = out
            preprocessing_time += video_preprocessing_time
            inference_time += video_inference_time
            processing_time += video_processing_time
            make_video(
                os.path.join(
                    args.output_dir, os.path.relpath(filename, args.videos_dir)
                ),
                decoded_images,
                format_code=args.output_format,
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
                1
                / (
                    avg_postprocessing_time
                    + avg_inference_time
                    + avg_preprocessing_time
                )
            )
        )
    else:
        cap = cv2.VideoCapture(0)
        if args.use_tvm:
            process_capture(model, cap, module=module)
        else:
            process_capture(model, cap)


if __name__ == "__main__":
    main()
