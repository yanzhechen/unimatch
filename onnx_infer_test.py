import onnxruntime as ort
import torch
import numpy as np
from utils import frame_utils
import cv2

def to_numpy(tensor):
    if tensor.dim() == 3:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.detach().cpu().numpy()

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 5,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
]
sess_options = ort.SessionOptions()
sess = ort.InferenceSession("unimatch_419_fit_dog.onnx", sess_options=sess_options, providers=providers)

set1 = "/mnt/sh_flex_storage/home/yanzhech/code/yz_unimatch/datasets/Unimatch_validation_data/set1/"
img1 = frame_utils.read_gen(set1+"frame0010_cam00.jpg")
# img1 = cv2.resize(img1, (384, 512), 
#                 interpolation = cv2.INTER_LINEAR)

img2 = frame_utils.read_gen(set1+"frame0010_cam01.jpg")
# img2 = cv2.resize(img2, (384, 512), 
#                 interpolation = cv2.INTER_LINEAR)
img1 = np.array(img1).astype(np.float32)[..., :3]
img2 = np.array(img2).astype(np.float32)[..., :3]

img1 = to_numpy(torch.from_numpy(img1).permute(2, 1, 0).float().unsqueeze(0))
img2 = to_numpy(torch.from_numpy(img2).permute(2, 1, 0).float().unsqueeze(0))

out_flow = sess.run(None, {"input1":img1,"input2": img2})
out = out_flow[0][-1].squeeze()
print(out)
print(out.shape)
out = np.transpose(out ,(2, 1, 0))
print(out.shape)
from utils import flow_viz
#flow_img = flow_viz.flow_to_color(out)
# flow_img = flow_viz.flow_to_image((out))

flow_viz.save_vis_flow_tofile(out, "test1-onnx.jpeg")
