import openvino as ov
import torch
import numpy as np
from utils import frame_utils
import cv2
import argparse
import os

from utils import flow_viz

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--device', '-d', default='cuda', type=str,
                        help='which device to run onnx infer, options: CPU')
    parser.add_argument('--out_file', '-out', default='test-ov.jpeg', type=str,
                        help='where to save the output jpg')
    parser.add_argument('--model', '-m', default='test_ovc.xml', type=str,
                        help='onnx model path')
    parser.add_argument("--input_path", '-ip', default='./datasets/Unimatch_validation_data/set1', type=str,
                        help='directory where is the input file')
    return parser

def to_numpy(tensor):
    if tensor.dim() == 3:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.detach().cpu().numpy()


def main(args):

    core = ov.Core()

    compiled_model = core.compile_model("test_ovc.xml", "AUTO")

    for i in range(9):

        set1 = args.input_path
        img1 = frame_utils.read_gen(os.path.join(set1, f"frame0010_cam0{i}.jpg"))
        img2 = frame_utils.read_gen(os.path.join(set1, f"frame0010_cam0{i+1}.jpg"))

        img1 = np.array(img1).astype(np.float32)[..., :3]
        img2 = np.array(img2).astype(np.float32)[..., :3]

        img1 = to_numpy(torch.from_numpy(img1).permute(2, 1, 0).float().unsqueeze(0))

        img2 = to_numpy(torch.from_numpy(img2).permute(2, 1, 0).float().unsqueeze(0))

        out = compiled_model([img1, img2])

        out = out[0][-1].squeeze()

        out = np.transpose(out ,(2, 1, 0))


        flow_viz.save_vis_flow_tofile(out, f"{args.out_file}{i}.jpg")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
