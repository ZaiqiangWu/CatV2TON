import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from tqdm import tqdm

from modules.cloth_masker import AutoMasker, vis_mask
from modules.pipeline import V2TONPipeline
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor


def read_video_frames(video_path, start_frame, end_frame, normalize=True):
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    video = read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
    if end_frame > video.size(0):  # end_frame is exclusive
        end_frame = video.size(0)
    video = video[start_frame:end_frame].permute(1, 0, 2, 3).unsqueeze(0)
    video = video.float() / 255.0
    if normalize:
        video = video * 2 - 1
    return video

class VideoDataset(Dataset):
    def __init__(self,video_path,cloth_type,clip_length=8):
        self.video_loader = MultithreadVideoLoader(video_path)
        self.clip_length = clip_length
        self.cloth_type = cloth_type#'upper', 'lower', 'overall', 'inner', 'outer'

        repo_path = snapshot_download(repo_id='zhengchong/CatVTON')
        self.automasker = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"),
            device='cuda',
        )
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                           do_convert_grayscale=True)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # x*2 - 1
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, id):
        # target shape [B, C, T, H, W]
        person_image = self.video_loader.cap()
        rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        person_image = Image.fromarray(rgb_image)
        person_image = resize_and_crop(person_image, (384, 512))
        preprocess =  self.automasker(
            person_image,
            self.cloth_type,
            densepose_colormap=cv2.COLORMAP_VIRIDIS
        )
        mask =preprocess['mask']
        mask = self.mask_processor.blur(mask, blur_factor=9)
        densepose = preprocess['densepose']
        masked_person = vis_mask(person_image, mask)
        masked_person = self.image_transforms(masked_person).unsqueeze(1).unsqueeze(0)#1CTHW
        mask = self.mask_transforms(mask).unsqueeze(1).unsqueeze(0)
        #print(np.array(densepose).shape)
        densepose = self.image_transforms(densepose).unsqueeze(1).unsqueeze(0)
        return masked_person, mask, densepose

    def gen_clip(self):
        persons=[]
        masks=[]
        denseposes=[]
        for i in range(self.clip_length):
            person, mask, densepose = self.__getitem__(i)
            persons.append(person)
            masks.append(mask)
            denseposes.append(densepose)
        persons=torch.cat(persons, dim=2)
        masks=torch.cat(masks, dim=2)
        masks=masks.repeat(1,3,1,1,1)
        denseposes=torch.cat(denseposes, dim=2)
        print(persons.shape)
        print(masks.shape)
        print(denseposes.shape)
        return persons, masks, denseposes



    def __len__(self):
        return len(self.video_loader)

 
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="alibaba-pai/EasyAnimateV4-XL-2-InP",
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="zhengchong/CatV2TON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--catvton_ckpt_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of CatVTON."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vivid",
        choices=["vivid", "vvt"],
        help="The dataset to evaluate the model on.",
    )
    parser.add_argument(
        "--data_root_path", 
        type=str, 
        help="Path to the dataset to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible evaluation."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for evaluation."
    )
      
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to perform.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.,
        help="The scale of classifier-free guidance for inference.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true",
        default=True,
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--load_pose",
        action="store_true", 
        default=True,
        help="Whether to load the pose video."
    )
    parser.add_argument(
        "--use_adacn",
        action="store_true",
        default=True,
        help="Whether to use AdaCN."
    )   
    parser.add_argument(
        "--slice_frames",
        type=int,
        default=24,
        help="The number of frames to slice the video into."
    )
    parser.add_argument(
        "--pre_frames",
        type=int,
        default=8,
        help="The number of frames to preprocess the video."
    )
    parser.add_argument(
        "--eval_pair",
        action="store_true",
        help="Whether or not to evaluate the pair.",
    )
    parser.add_argument(
        "--concat_eval_results",
        action="store_true",
        help="Whether or not to  concatenate the all conditions into one image.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

from einops import rearrange


def repaint(
    person: torch.Tensor,
    mask: torch.Tensor,
    result: torch.Tensor,
    kernal_size: int=None,
    ):
    if kernal_size is None:
        h = person.size(-1)
        kernal_size = h // 50
        if kernal_size % 2 == 0:
            kernal_size += 1
    # Apply 2D average pooling on mask video
    # (B, C, F, H, W) -> (B*F, C, H, W)
    mask = rearrange(mask, 'b c f h w -> (b f) c h w')
    mask = torch.nn.functional.avg_pool2d(mask, kernal_size, stride=1, padding=kernal_size // 2)
    mask = rearrange(mask, '(b f) c h w -> b c f h w', b=person.size(0))
    # Use mask video to repaint result video
    result = person * (1 - mask) + result * mask
    return result

@torch.no_grad()
def main():
    args = parse_args()
    
    # Pipeline
    base_model_path = snapshot_download(args.base_model_path) if not os.path.exists(args.base_model_path) else args.base_model_path
    finetuned_model_path = snapshot_download(args.finetuned_model_path) if not os.path.exists(args.finetuned_model_path) else args.finetuned_model_path
    finetuned_model_path = os.path.join(finetuned_model_path, "512-64K" if True else "256-128K")
    pipeline = V2TONPipeline(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        load_pose=args.load_pose,
        torch_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
    )
    
    # Dataset

    video_path = './videos/jin_16_test.mp4'
    cloth_path = './garments/upperbody/jin_00_white_bg.jpg'
    video_dataset = VideoDataset(video_path, 'upper', clip_length=8)
    cloth_image = Image.open(cloth_path).convert("RGB")
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    cloth_image = video_dataset.image_transforms(cloth_image).unsqueeze(1).unsqueeze(0)









    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    gt_path = os.path.join(args.output_dir, f"{args.dataset}-{args.height}", "gt")
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset}-{args.height}", "paired" if args.eval_pair else "unpaired")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for i in range(1):
        persons, masks, denseposes = video_dataset.gen_clip()
        
        # Inference
        results = pipeline.video_try_on(
            source_video=persons,
            condition_image=cloth_image,
            mask_video=masks,
            pose_video=denseposes, #if args.load_pose else None,
            num_inference_steps=args.num_inference_steps,
            slice_frames=args.slice_frames,
            pre_frames=args.pre_frames,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_adacn=args.use_adacn
        )  # [B, F, H, W, C]
        results = results.permute(0, 4, 1, 2, 3)  # [B, C, F, H, W]
        
        # Repaint
        if args.repaint:
            results = repaint(persons, masks, results)
        print(results.shape)

            

if __name__ == "__main__":
    main()