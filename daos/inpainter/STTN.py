# from PIL import Image
# import os
# import time
# import copy
# import cv2
# import numpy as np
# import torch
# from typing import List, Optional, Tuple
# from torchvision import transforms

# from daos.inpainter.utils.sttn_utils import Stack, ToTorchFormatTensor
# from utils.config import Config
# from utils.logger import setup_logger


# class STTN:
#   def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#     self.logger = setup_logger('STTN')
#     self.config = Config.get_config()
#     self.device = device

#     model_path = os.path.join(
#         self.config['models']['BASE']['Inpainter'],
#         self.config['models']['PATH']['Inpainter']['STTN']
#     )

#     self.model = torch.jit.load(
#         model_path, map_location=self.device).to(self.device)
#     self.model.eval()

#     self.neighbor_stride = self.config.STTN_NEIGHBOR_STRIDE or 5
#     self.ref_length = self.config.STTN_REFERENCE_LENGTH

#     self._to_tensors = transforms.Compose([
#         Stack(),
#         ToTorchFormatTensor()
#     ])

#   @torch.no_grad()
#   def predict(self, video_path: str, mask_dir: str, output_path: str = "output.mp4", fps: int = 30):
#     # Get dimensions from first mask image
#     first_mask = Image.open(os.path.join(
#         mask_dir, sorted(os.listdir(mask_dir))[0]))
#     w, h = first_mask.size

#     ref_length = self.ref_length
#     neighbor_stride = self.neighbor_stride

#     def get_ref_index(neighbor_ids, length):
#       return [i for i in range(0, length, ref_length) if i not in neighbor_ids]

#     def read_frame_from_video(vpath):
#       frames = []
#       cap = cv2.VideoCapture(vpath)
#       while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#           break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(Image.fromarray(frame).resize((w, h)))
#       cap.release()
#       return frames

#     def read_masks(mdir):
#       mask_imgs = []
#       filenames = sorted(os.listdir(mdir))
#       for fname in filenames:
#         mask = Image.open(os.path.join(mdir, fname)).convert(
#             'L').resize((w, h), Image.NEAREST)
#         mask = np.array(mask) > 0
#         mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
#             cv2.MORPH_CROSS, (3, 3)), iterations=4)
#         mask_imgs.append(Image.fromarray(mask * 255))
#       return mask_imgs

#     frames_pil = read_frame_from_video(video_path)
#     video_length = len(frames_pil)
#     feats = self._to_tensors(frames_pil).unsqueeze(0).to(self.device) * 2 - 1
#     orig_frames = [np.array(f).astype(np.uint8) for f in frames_pil]

#     masks_pil = read_masks(mask_dir)
#     binary_masks = [np.expand_dims(
#         (np.array(m) != 0).astype(np.uint8), 2) for m in masks_pil]
#     masks = self._to_tensors(masks_pil).unsqueeze(0).to(self.device)

#     feats = self.model.encoder(
#         (feats * (1 - masks)).view(video_length, 3, h, w))
#     _, c, feat_h, feat_w = feats.size()
#     feats = feats.view(1, video_length, c, feat_h, feat_w)

#     comp_frames = [None] * video_length

#     for f in range(0, video_length, neighbor_stride):
#       neighbor_ids = list(range(max(0, f - neighbor_stride),
#                           min(video_length, f + neighbor_stride + 1)))
#       ref_ids = get_ref_index(neighbor_ids, video_length)

#       pred_feat = self.model.infer(
#           feats[0, neighbor_ids + ref_ids], masks[0, neighbor_ids + ref_ids])
#       pred_img = torch.tanh(self.model.decoder(
#           pred_feat[:len(neighbor_ids)])).detach()
#       pred_img = (pred_img + 1) / 2
#       pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255

#       for i, idx in enumerate(neighbor_ids):
#         inpainted = pred_img[i].astype(np.uint8)
#         mask = binary_masks[idx]
#         img = inpainted * mask + orig_frames[idx] * (1 - mask)
#         if comp_frames[idx] is None:
#           comp_frames[idx] = img
#         else:
#           comp_frames[idx] = (comp_frames[idx].astype(
#               np.float32) * 0.5 + img.astype(np.float32) * 0.5)

#     out = cv2.VideoWriter(
#         output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#     for idx in range(video_length):
#       final = comp_frames[idx].astype(
#           np.uint8) * binary_masks[idx] + orig_frames[idx] * (1 - binary_masks[idx])
#       bgr_frame = cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_RGB2BGR)
#       out.write(bgr_frame)
#     out.release()
#     self.logger.info(f"Video saved to {output_path}")
