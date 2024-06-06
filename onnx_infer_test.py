import onnxruntime as ort
import torch
import numpy as np
from utils import frame_utils
import cv2
import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--device', '-d', default='cuda', type=str,
                        help='which device to run onnx infer, options: CPU')
    parser.add_argument('--out_file', '-out', default='test-onnx.jpeg', type=str,
                        help='where to save the output jpg')
    parser.add_argument('--model', '-m', default='unimatch_419_fit_dog.onnx', type=str,
                        help='where to save the output jpg')
    parser.add_argument("--input_path", '-ip', default='.', type=str,
                        help='directory where is the input file')
    return parser

def to_numpy(tensor):
    if tensor.dim() == 3:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.detach().cpu().numpy()


def main(args):

    sess_options = ort.SessionOptions()
    if args.device.upper() == 'CPU':
        sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    elif args.device.upper() == 'CUDA':
        cuda_providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
        ]
        sess = ort.InferenceSession(args.model, sess_options=sess_options, providers=cuda_providers)
    elif args.device.upper() == 'XPU':
        sess = ort.InferenceSession(args.model, providers=['OpenVINOExecutionProvider'], provider_options=[{"device_type": "GPU.1_FP32"}])


    set1 = args.input_path
    img1 = frame_utils.read_gen(os.path.join(set1, "frame0010_cam00.jpg"))
    img2 = frame_utils.read_gen(os.path.join(set1, "frame0010_cam01.jpg"))

    img1 = np.array(img1).astype(np.float32)[..., :3]
    img2 = np.array(img2).astype(np.float32)[..., :3]

    img1 = to_numpy(torch.from_numpy(img1).permute(2, 1, 0).float().unsqueeze(0))
    img2 = to_numpy(torch.from_numpy(img2).permute(2, 1, 0).float().unsqueeze(0))

    out_flow = sess.run(None, {"input1":img1,"input2": img2})
    out = out_flow[0][-1].squeeze()

    out = np.transpose(out ,(2, 1, 0))

    from utils import flow_viz

    flow_viz.save_vis_flow_tofile(out, args.out_file)



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
