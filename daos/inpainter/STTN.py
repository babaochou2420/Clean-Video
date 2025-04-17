import os
import time
import copy
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
from torchvision import transforms

from daos.inpainter.STTN import InpaintGenerator
from daos.inpainter.utils.sttn_utils import Stack, ToTorchFormatTensor
from utils.config import Config
from utils.logger import setup_logger


class STTN:
  def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    self.logger = setup_logger('STTN')
    self.config = Config.getConfig()
    self.device = device
    self.model = InpaintGenerator().to(self.device)

    self.modelPath = os.path.join(
        self.config['models']['BASE']['Inpainter'], self.config['models']['PATH']['Inpainter']['STTN'])

    self.model.load_state_dict(torch.load(
        self.modelPath, map_location=self.device)['netG'])
    self.model.eval()

    # Make Dynamic
    # self.model_input_width = 640
    # self.model_input_height = self.config.STTN_MODEL_INPUT_HEIGHT
    self.neighbor_stride = self.config.STTN_NEIGHBOR_STRIDE or 5
    self.ref_length = self.config.STTN_REFERENCE_LENGTH

    self._to_tensors = transforms.Compose([
        Stack(),
        ToTorchFormatTensor()
    ])

  def inpaint_video(self, video_path: str, mask_path: Optional[str] = None, clip_gap: Optional[int] = None, output_path: Optional[str] = None) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise IOError(f"Failed to open video: {video_path}")

    frame_info = {
        'W_ori': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'H_ori': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'len': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    output_path = output_path or os.path.splitext(video_path)[
        0] + '_inpainted.mp4'
    clip_gap = clip_gap or self.config.STTN_MAX_LOAD_NUM

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_info['fps'],
        (frame_info['W_ori'], frame_info['H_ori'])
    )

    mask = self._load_mask(mask_path, frame_info['H_ori'], frame_info['W_ori'])
    split_h = int(frame_info['W_ori'] * 3 / 16)
    inpaint_areas = self._get_inpaint_area_by_mask(
        frame_info['H_ori'], split_h, mask)

    for i in range((frame_info['len'] + clip_gap - 1) // clip_gap):
      start_f = i * clip_gap
      end_f = min((i + 1) * clip_gap, frame_info['len'])
      frames_hr = []
      frames = {k: [] for k in range(len(inpaint_areas))}

      for j in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
          break
        frames_hr.append(frame)
        for k, (top, bottom) in enumerate(inpaint_areas):
          cropped = frame[top:bottom, :, :]
          resized = cv2.resize(
              cropped, (self.model_input_width, self.model_input_height))
          frames[k].append(resized)

      comps = {k: self._inpaint(frames[k]) for k in frames}

      for j in range(len(frames_hr)):
        frame = frames_hr[j]
        for k, (top, bottom) in enumerate(inpaint_areas):
          comp = cv2.resize(comps[k][j], (frame_info['W_ori'], split_h))
          comp = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
          mask_area = mask[top:bottom, :, :]
          frame[top:bottom, :, :] = mask_area * comp + \
              (1 - mask_area) * frame[top:bottom, :, :]
        writer.write(frame)

    writer.release()
    cap.release()
    return output_path

  def _load_mask(self, path: Optional[str], height: int, width: int) -> np.ndarray:
    if path is None:
      raise ValueError("Mask path must be provided.")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return mask[:, :, None]

  def _inpaint(self, frames: List[np.ndarray]) -> List[np.ndarray]:
    feats = self._to_tensors(frames).unsqueeze(0).to(self.device) * 2 - 1
    frame_length = len(frames)

    with torch.no_grad():
      feats = self.model.encoder(feats.view(
          frame_length, 3, self.model_input_height, self.model_input_width))
      _, c, h, w = feats.shape
      feats = feats.view(1, frame_length, c, h, w)

    comp_frames = [None] * frame_length

    for f in range(0, frame_length, self.neighbor_stride):
      neighbor_ids = list(range(max(0, f - self.neighbor_stride),
                          min(frame_length, f + self.neighbor_stride + 1)))
      ref_ids = self._get_ref_index(neighbor_ids, frame_length)

      with torch.no_grad():
        pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids])
        pred_img = torch.tanh(self.model.decoder(
            pred_feat[:len(neighbor_ids)])).detach()
        pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255

      for i, idx in enumerate(neighbor_ids):
        img = pred_img[i].astype(np.uint8)
        if comp_frames[idx] is None:
          comp_frames[idx] = img
        else:
          comp_frames[idx] = ((comp_frames[idx].astype(
              np.float32) + img.astype(np.float32)) * 0.5).astype(np.uint8)

    return comp_frames

  def _get_ref_index(self, neighbor_ids: List[int], total_length: int) -> List[int]:
    return [i for i in range(0, total_length, self.ref_length) if i not in neighbor_ids]

  def _get_inpaint_area_by_mask(self, H: int, h: int, mask: np.ndarray) -> List[Tuple[int, int]]:
    inpaint_area = []
    to_H = from_H = H
    while from_H != 0:
      if to_H - h < 0:
        from_H = 0
        to_H = h
      else:
        from_H = to_H - h
      if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
        if to_H != H:
          move = 0
          while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
            move += 1
          if to_H + move < H and move < h:
            to_H += move
            from_H += move
        if (from_H, to_H) not in inpaint_area:
          inpaint_area.append((from_H, to_H))
        else:
          break
      to_H -= h
    return inpaint_area
