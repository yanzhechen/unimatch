import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os


from dataloader.flow.datasets import build_train_dataset, FlyingChairs, FlyingThings3D, MpiSintel, KITTI
from unimatch.unimatch import UniMatch
from loss.flow_loss import flow_loss_func

from evaluate_flow import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                           create_kitti_submission, create_sintel_submission,
                           inference_flow,
                           )

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation datasets')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')

    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')
    parser.add_argument('--val_things_clean_only', action='store_true')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed methic when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo if only visualization is needed')

    # inference on images or videos
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_video', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size for the input to the network')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--pred_bwd_flow', action='store_true',
                        help='predict backward flow only')
    parser.add_argument('--fwd_bwd_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--concat_flow_img', action='store_true')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # misc
    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time')

    parser.add_argument('--debug', action='store_true')

    return parser


def main(args):
    print_info = not args.eval and not args.submission and args.inference_dir is None and args.inference_video is None

    if print_info and args.local_rank == 0:
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)

    misc.check_path(args.output_path)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True


    args.distributed = False
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # model
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    if print_info:
        num_params = sum(p.numel() for p in model.parameters())
        
        print(model)
        print(num_params)

    if 0 & torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    
    if print_info:
        print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None and args.inference_video is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        #loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        loc = 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        if print_info:
            print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
    #c_model = nncf.compress_weights(model)
    #print(c_model)
    # evaluate
    import nncf
    import logging
    #nncf.set_log_level(logging)
    val_dataset = FlyingChairs(split='validation')
    #val_dataset = MpiSintel(split='training')
    print(val_dataset)
    print(len(val_dataset.image_list))
    def transform_fn(data_item):
        image1, image2, _, _ = data_item
        return image1, image2

    # Step 2: Initialize NNCF Dataset
    calibration_dataset = nncf.Dataset(val_dataset, transform_fn)
    # Step 3: Run the quantization pipeline
    # quantized_model = nncf.quantize(model, calibration_dataset, subset_size=40)
    import openvino as ov

    # Flying chairs
    dummy_input1 = torch.randn(1, 3, 384, 512)
    dummy_input2 = torch.randn(1, 3, 384, 512)
    # ov_model = ov.convert_model(model.cpu(), input=[("img0", [1, 3, 384,512]), ("img1", [1, 3, 384,512])], example_input=(dummy_input1,dummy_input2))

    # ov.save_model(ov_model, 'ori_unimatch_3r.xml', compress_to_fp16=False)
    # exit()
    # Sintel
#    dummy_input1 = torch.randn(1, 3, 436, 1024)
#    dummy_input2 = torch.randn(1, 3, 436, 1024)
    torch.onnx.export(model,               # model being run
                 (dummy_input1, dummy_input2),                         # model input (or a tuple for multiple inputs)
                 "unimatch_batch_quantized.onnx",   # where to save the model (can be a file or file-like object)
                 export_params=True,        # store the trained parameter weights inside the model file
                 opset_version=16,          # the ONNX version to export the model to
                 do_constant_folding=True,  # whether to execute constant folding for optimization
                 input_names = ['input1', 'input2'],   # the model's input names
                 output_names = ['output'], # the model's output names
               #   dynamic_axes={'input1' : {0 : 'batch_size'},
               #                 'input2' : {0 : 'batch_size'},# variable length axes
               #                 'output' : {0 : 'batch_size'}}
                 )
    print("export model to onnx compeleted")
    exit()
    ov_quantized_model = ov.convert_model(quantized_model.cpu(), input=[("img0", [1, 3, 384,512]), ("img1", [1, 3, 384,512])], example_input=(dummy_input1,dummy_input2))
#    ov_quantized_model = ov.convert_model(quantized_model.cpu(), input=[("img0", [1, 3, 436,1024]), ("img1", [1, 3, 436,1024])], example_input=(dummy_input1,dummy_input2))
    ov.save_model(ov_quantized_model, 'quantized_unimatch_whole_cali_wo_reg.xml', compress_to_fp16=True)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)

