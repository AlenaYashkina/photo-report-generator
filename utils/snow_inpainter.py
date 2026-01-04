"""Snow inpainting helper for ground/soil areas in photos."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    torch = None
    StableDiffusionInpaintPipeline = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ENCODING_MAP = {
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".png": ".png",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tif",
    ".webp": ".webp",
}


@dataclass(frozen=True)
class SnowInpaintConfig:
    snow_s_max: int = 60
    snow_v_min: int = 180
    snow_s_max_warm: int = 110
    snow_v_min_warm: int = 190
    snow_l_min: int = 190
    snow_chroma_max: int = 60
    snow_chroma_seed_max: int = 35
    snow_color_sigma: float = 2.4
    snow_ratio_threshold: float = 0.03

    soil_h_min: int = 5
    soil_h_max: int = 35
    soil_s_min: int = 50
    soil_v_min: int = 20
    soil_v_max: int = 200
    soil_s_neutral_max: int = 80
    soil_v_neutral_min: int = 25
    soil_v_neutral_max: int = 200
    dark_v_max: int = 90
    dark_s_min: int = 30
    dark_h_max: int = 45
    bright_s_min: int = 120
    bright_v_min: int = 210
    soil_color_sigma: float = 2.8
    soil_chroma_max: int = 90
    min_soil_pixels: int = 500
    color_std_min: float = 6.0

    bottom_ratio: float = 0.7
    aggressive_dilate_ratio: float = 0.006
    ring_ratio: float = 0.012
    inpaint_radius_ratio: float = 0.01
    mask_cleanup_ratio: float = 0.006
    mask_blur_ratio: float = 0.008
    texture_blur_ratio: float = 0.004
    component_min_area_ratio: float = 0.0015
    ground_k: int = 3
    ground_scale: float = 0.25
    ground_bottom_ratio: float = 0.6
    ground_min_area_ratio: float = 0.05
    ground_min_bottom_fraction: float = 0.25
    ground_y_weight: float = 40.0
    ground_cleanup_ratio: float = 0.01
    ground_dist_quantile: float = 0.9
    ground_dist_margin: float = 0.12
    ground_touch_ratio: float = 0.12

    seed_alpha: float = 0.7
    overlay_alpha: float = 0.35
    overlay_alpha_strong: float = 0.5

    noise_sigma_min: float = 6.0
    noise_sigma_max: float = 18.0
    min_snow_pixels: int = 500


@dataclass(frozen=True)
class SDInpaintConfig:
    model_id: str = ""
    device: str = "auto"
    steps: int = 28
    cfg_scale: float = 7.0
    strength: float = 0.55
    strength_strong: float = 0.7
    seed: Optional[int] = None
    mask_blur_ratio: float = 0.006
    prompt: str = (
        "photorealistic snow covering soil, winter ground, natural snow texture, "
        "realistic lighting, detailed"
    )
    prompt_strong: str = (
        "thick snow cover on ground, fresh snow, photorealistic, natural texture, detailed"
    )
    negative_prompt: str = (
        "blurry, lowres, artifacts, plastic, overexposed, oversaturated, "
        "cartoon, painting, text, watermark"
    )
    xformers: bool = False
    attention_slicing: bool = True


def _kernel_size(height: int, width: int, ratio: float, min_size: int = 3, max_size: int = 101) -> int:
    size = int(round(min(height, width) * ratio))
    size = max(min_size, min(max_size, size))
    if size % 2 == 0:
        size += 1
    return size


def _cleanup_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def _apply_vertical_bias(mask: np.ndarray, bottom_ratio: float) -> np.ndarray:
    if bottom_ratio >= 1.0:
        return mask
    height = mask.shape[0]
    cutoff = int(round(height * (1.0 - bottom_ratio)))
    if cutoff > 0:
        mask[:cutoff, :] = 0
    return mask


def _mask_ratio(mask: np.ndarray) -> float:
    total = mask.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total)


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    kept = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == idx] = 255
    return kept


def _keep_components_touching_bottom(mask: np.ndarray, bottom_ratio: float) -> np.ndarray:
    if bottom_ratio <= 0:
        return mask
    height, width = mask.shape[:2]
    bottom_start = int(round(height * (1.0 - bottom_ratio)))
    bottom_start = max(0, min(bottom_start, height - 1))
    bottom = np.zeros((height, width), dtype=bool)
    bottom[bottom_start:, :] = True

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    kept = np.zeros_like(mask)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] <= 0:
            continue
        component = labels == idx
        if np.any(component & bottom):
            kept[component] = 255
    return kept


def _compute_chroma(lab: np.ndarray) -> np.ndarray:
    a_delta = lab[:, :, 1].astype(np.float32) - 128.0
    b_delta = lab[:, :, 2].astype(np.float32) - 128.0
    return np.sqrt(a_delta * a_delta + b_delta * b_delta)


def _refine_color_mask(
    lab: np.ndarray,
    seed_mask: np.ndarray,
    base_mask: np.ndarray,
    sigma: float,
    min_pixels: int,
    std_min: float,
) -> Optional[np.ndarray]:
    if sigma <= 0:
        return None
    idx = seed_mask > 0
    count = int(np.count_nonzero(idx))
    if count < min_pixels:
        return None
    samples = lab[idx].reshape(-1, 3).astype(np.float32)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std = np.maximum(std, std_min)
    lab_f = lab.astype(np.float32)
    diff = (lab_f - mean) / std
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    refined = (dist <= sigma) & (base_mask > 0)
    return refined.astype(np.uint8) * 255


def _estimate_ground_mask(image: np.ndarray, cfg: SnowInpaintConfig) -> np.ndarray:
    height, width = image.shape[:2]
    bottom_start = int(round(height * (1.0 - cfg.bottom_ratio)))
    fallback = np.zeros((height, width), dtype=np.uint8)
    if bottom_start < height:
        fallback[bottom_start:, :] = 255

    scale = cfg.ground_scale
    if scale <= 0:
        return fallback
    small_w = min(width, max(32, int(round(width * scale))))
    small_h = min(height, max(32, int(round(height * scale))))
    if small_w < 2 or small_h < 2:
        return fallback

    try:
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    except cv2.error:
        return fallback

    bottom_start_small = int(round(small_h * (1.0 - cfg.ground_bottom_ratio)))
    bottom_start_small = max(0, min(bottom_start_small, small_h - 1))
    bottom_lab = lab[bottom_start_small:, :, :].reshape(-1, 3)
    if bottom_lab.shape[0] < max(32, cfg.ground_k):
        return fallback

    k = max(2, min(int(cfg.ground_k), 6))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(bottom_lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    total = float(len(labels)) if len(labels) else 1.0
    selected = [i for i, c in enumerate(counts) if (c / total) >= cfg.ground_min_area_ratio]
    if not selected:
        selected = [int(np.argmax(counts))]

    full_lab = lab.reshape(-1, 3)
    centers_sel = centers[selected]
    diffs = full_lab[:, None, :] - centers_sel[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    min_dist = np.min(dist, axis=1)

    bottom_mask = np.zeros((small_h, small_w), dtype=bool)
    bottom_mask[bottom_start_small:, :] = True
    bottom_dist = min_dist[bottom_mask.reshape(-1)]
    if bottom_dist.size == 0:
        return fallback
    threshold = float(np.quantile(bottom_dist, cfg.ground_dist_quantile))
    threshold = threshold * (1.0 + cfg.ground_dist_margin)

    ground_small = (min_dist <= threshold).reshape(small_h, small_w).astype(np.uint8) * 255
    ground = cv2.resize(ground_small, (width, height), interpolation=cv2.INTER_NEAREST)
    ground = _apply_vertical_bias(ground, cfg.bottom_ratio)
    cleanup = _kernel_size(height, width, cfg.ground_cleanup_ratio, max_size=81)
    ground = _cleanup_mask(ground, cleanup)
    ground = _keep_components_touching_bottom(ground, cfg.ground_touch_ratio)
    min_area = max(200, int(round(height * width * cfg.component_min_area_ratio)))
    ground = _filter_small_components(ground, min_area)
    return ground


def _read_image(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def _write_image(path: Path, image: np.ndarray) -> bool:
    suffix = ENCODING_MAP.get(path.suffix.lower(), ".jpg")
    output_path = path.with_suffix(suffix)
    try:
        ok, buffer = cv2.imencode(suffix, image)
    except cv2.error:
        return False
    if not ok:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(output_path))
    return True


def _choose_device(device: str) -> str:
    device = device.strip().lower() if device else "auto"
    if device == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise ValueError("CUDA requested but not available. Use --device cpu or install CUDA-enabled torch.")
    return device


def _prepare_sd_size(width: int, height: int, multiple: int = 8) -> Tuple[int, int]:
    new_w = (width + multiple - 1) // multiple * multiple
    new_h = (height + multiple - 1) // multiple * multiple
    return new_w, new_h


def _prepare_sd_image(arr: np.ndarray, target_size: Tuple[int, int], resample: int) -> Image.Image:
    image = Image.fromarray(arr)
    if image.size != target_size:
        image = image.resize(target_size, resample)
    return image


def _blur_mask(mask: np.ndarray, ratio: float, max_size: int = 81) -> np.ndarray:
    if ratio <= 0:
        return mask
    size = _kernel_size(mask.shape[0], mask.shape[1], ratio, min_size=1, max_size=max_size)
    if size <= 1:
        return mask
    return cv2.GaussianBlur(mask, (size, size), 0)


def _blend_with_mask(base: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or not np.any(mask):
        return patch
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = base.astype(np.float32) * (1.0 - mask_f) + patch.astype(np.float32) * mask_f
    return np.clip(blended, 0, 255).astype(np.uint8)


class SDInpainter:
    def __init__(self, cfg: SDInpaintConfig):
        if torch is None or StableDiffusionInpaintPipeline is None:
            raise SystemExit(
                "diffusers and torch are required. Install via `pip install torch diffusers transformers safetensors`."
            )
        if not cfg.model_id:
            raise ValueError("SD model is required. Use --sd-model or set SD_MODEL.")
        device = _choose_device(cfg.device)
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                cfg.model_id,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load SD model: {cfg.model_id}") from exc
        if cfg.attention_slicing:
            self.pipe.enable_attention_slicing()
        if cfg.xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.cfg = cfg

    def inpaint(self, image: Image.Image, mask: Image.Image, strong: bool) -> Image.Image:
        generator = None
        if self.cfg.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed))
        strength = self.cfg.strength_strong if strong else self.cfg.strength
        prompt = self.cfg.prompt_strong if strong else self.cfg.prompt
        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=image,
            mask_image=mask,
            strength=strength,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.cfg_scale,
            generator=generator,
        )
        return result.images[0]


def _detect_snow_mask(
    hsv: np.ndarray,
    lab: np.ndarray,
    chroma: np.ndarray,
    cfg: SnowInpaintConfig,
    ground_mask: Optional[np.ndarray],
) -> np.ndarray:
    _, s, v = cv2.split(hsv)
    l_chan = lab[:, :, 0]
    snow_hsv = (s <= cfg.snow_s_max) & (v >= cfg.snow_v_min)
    snow_warm = (s <= cfg.snow_s_max_warm) & (v >= cfg.snow_v_min_warm)
    snow_lab = l_chan >= cfg.snow_l_min
    snow = (snow_hsv | snow_warm | snow_lab) & (chroma <= cfg.snow_chroma_max)

    mask = snow.astype(np.uint8) * 255
    if ground_mask is not None:
        mask = cv2.bitwise_and(mask, ground_mask)

    seed = (snow_hsv | snow_lab) & (chroma <= cfg.snow_chroma_seed_max)
    seed_mask = seed.astype(np.uint8) * 255
    if ground_mask is not None:
        seed_mask = cv2.bitwise_and(seed_mask, ground_mask)

    refined = _refine_color_mask(
        lab,
        seed_mask,
        mask,
        cfg.snow_color_sigma,
        cfg.min_snow_pixels,
        cfg.color_std_min,
    )
    if refined is not None:
        mask = refined

    size = _kernel_size(lab.shape[0], lab.shape[1], cfg.mask_cleanup_ratio, max_size=31)
    mask = _cleanup_mask(mask, size)
    min_area = max(200, int(round(lab.shape[0] * lab.shape[1] * cfg.component_min_area_ratio)))
    return _filter_small_components(mask, min_area)


def _detect_soil_mask(
    hsv: np.ndarray,
    lab: np.ndarray,
    chroma: np.ndarray,
    snow_mask: Optional[np.ndarray],
    cfg: SnowInpaintConfig,
    ground_mask: Optional[np.ndarray],
) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    brown = (
        (h >= cfg.soil_h_min)
        & (h <= cfg.soil_h_max)
        & (s >= cfg.soil_s_min)
        & (v >= cfg.soil_v_min)
        & (v <= cfg.soil_v_max)
    )
    neutral = (
        (s <= cfg.soil_s_neutral_max)
        & (v >= cfg.soil_v_neutral_min)
        & (v <= cfg.soil_v_neutral_max)
    )
    dark = (v <= cfg.dark_v_max) & (s >= cfg.dark_s_min) & (h <= cfg.dark_h_max)
    bright_color = (v >= cfg.bright_v_min) & (s >= cfg.bright_s_min)
    soil = (brown | dark | neutral) & (~bright_color) & (chroma <= cfg.soil_chroma_max)
    if snow_mask is not None:
        soil &= snow_mask == 0

    mask = soil.astype(np.uint8) * 255
    if ground_mask is not None:
        mask = cv2.bitwise_and(mask, ground_mask)

    seed = soil & (v >= cfg.soil_v_min) & (v <= cfg.soil_v_max)
    seed_mask = seed.astype(np.uint8) * 255
    if ground_mask is not None:
        seed_mask = cv2.bitwise_and(seed_mask, ground_mask)

    refined = _refine_color_mask(
        lab,
        seed_mask,
        mask,
        cfg.soil_color_sigma,
        cfg.min_soil_pixels,
        cfg.color_std_min,
    )
    if refined is not None:
        mask = refined

    mask = _apply_vertical_bias(mask, cfg.bottom_ratio)
    size = _kernel_size(lab.shape[0], lab.shape[1], cfg.mask_cleanup_ratio, max_size=41)
    mask = _cleanup_mask(mask, size)
    min_area = max(200, int(round(lab.shape[0] * lab.shape[1] * cfg.component_min_area_ratio)))
    return _filter_small_components(mask, min_area)


def _sample_snow_stats(image: np.ndarray, snow_mask: np.ndarray, min_pixels: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if snow_mask is None:
        return None
    idx = snow_mask > 0
    if int(np.count_nonzero(idx)) < min_pixels:
        return None
    pixels = image[idx]
    if pixels.size == 0:
        return None
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    return mean, std


def _make_snow_texture(
    shape: Tuple[int, int, int],
    rng: np.random.Generator,
    stats: Optional[Tuple[np.ndarray, np.ndarray]],
    cfg: SnowInpaintConfig,
) -> np.ndarray:
    height, width = shape[:2]
    if stats:
        mean, std = stats
        base = mean.astype(np.float32)
        sigma = np.clip(std.astype(np.float32), cfg.noise_sigma_min, cfg.noise_sigma_max)
    else:
        base = np.array([235.0, 235.0, 235.0], dtype=np.float32)
        sigma = np.array([cfg.noise_sigma_max] * 3, dtype=np.float32)
    noise = rng.normal(0.0, sigma, (height, width, 3)).astype(np.float32)
    texture = base + noise
    texture = np.clip(texture, 200, 255).astype(np.uint8)
    blur = _kernel_size(height, width, cfg.texture_blur_ratio, max_size=31)
    texture = cv2.GaussianBlur(texture, (blur, blur), 0)
    return texture


def _apply_overlay(
    image: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    blur_size: int,
) -> np.ndarray:
    if mask is None or not np.any(mask):
        return image
    if blur_size > 1:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = image.astype(np.float32) * (1.0 - alpha * mask_f) + overlay.astype(np.float32) * (alpha * mask_f)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _ring_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel)
    return cv2.subtract(dilated, mask)


def _strengthen_existing_snow(
    image: np.ndarray,
    snow_mask: np.ndarray,
    texture: np.ndarray,
    cfg: SnowInpaintConfig,
) -> np.ndarray:
    blur = _kernel_size(image.shape[0], image.shape[1], cfg.mask_blur_ratio, max_size=41)
    return _apply_overlay(image, texture, snow_mask, cfg.overlay_alpha_strong, blur)


def _inpaint_snow(
    image: np.ndarray,
    soil_mask: np.ndarray,
    snow_mask: np.ndarray,
    cfg: SnowInpaintConfig,
) -> np.ndarray:
    snow_ratio = _mask_ratio(snow_mask) if snow_mask is not None else 0.0
    aggressive = snow_ratio >= cfg.snow_ratio_threshold

    if aggressive and soil_mask is not None and np.any(soil_mask):
        size = _kernel_size(image.shape[0], image.shape[1], cfg.aggressive_dilate_ratio, max_size=61)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        soil_mask = cv2.dilate(soil_mask, kernel)

    rng = np.random.default_rng()
    stats = _sample_snow_stats(image, snow_mask, cfg.min_snow_pixels)
    texture = _make_snow_texture(image.shape, rng, stats, cfg)

    if soil_mask is None or not np.any(soil_mask):
        if snow_mask is not None and np.any(snow_mask):
            return _strengthen_existing_snow(image, snow_mask, texture, cfg)
        return image

    seed = image.copy()
    ring_size = _kernel_size(image.shape[0], image.shape[1], cfg.ring_ratio, max_size=61)
    ring = _ring_mask(soil_mask, ring_size)
    seed = _apply_overlay(seed, texture, ring, cfg.seed_alpha, blur_size=ring_size)

    inpaint_radius = _kernel_size(
        image.shape[0],
        image.shape[1],
        cfg.inpaint_radius_ratio,
        min_size=3,
        max_size=31,
    )
    inpainted = cv2.inpaint(seed, soil_mask, inpaint_radius, cv2.INPAINT_TELEA)

    overlay_mask = soil_mask
    overlay_alpha = cfg.overlay_alpha
    if aggressive and snow_mask is not None and np.any(snow_mask):
        overlay_mask = cv2.bitwise_or(soil_mask, snow_mask)
        overlay_alpha = cfg.overlay_alpha_strong

    blur = _kernel_size(image.shape[0], image.shape[1], cfg.mask_blur_ratio, max_size=41)
    return _apply_overlay(inpainted, texture, overlay_mask, overlay_alpha, blur)


def _prepare_sd_mask(
    image: np.ndarray,
    soil_mask: Optional[np.ndarray],
    snow_mask: Optional[np.ndarray],
    cfg: SnowInpaintConfig,
) -> Tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    target = soil_mask.copy() if soil_mask is not None else np.zeros((height, width), dtype=np.uint8)

    snow_mask_sd = None
    if snow_mask is not None:
        snow_mask_sd = snow_mask.copy()
        snow_mask_sd = _apply_vertical_bias(snow_mask_sd, cfg.bottom_ratio)
        cleanup = _kernel_size(height, width, cfg.mask_cleanup_ratio, max_size=41)
        snow_mask_sd = _cleanup_mask(snow_mask_sd, cleanup)

    snow_pixels = int(np.count_nonzero(snow_mask_sd)) if snow_mask_sd is not None else 0
    snow_ratio = snow_pixels / float(height * width) if height and width else 0.0
    strong = snow_pixels >= cfg.min_snow_pixels or snow_ratio >= cfg.snow_ratio_threshold

    if strong and snow_mask_sd is not None:
        target = cv2.bitwise_or(target, snow_mask_sd)

    target = _apply_vertical_bias(target, cfg.bottom_ratio)
    cleanup = _kernel_size(height, width, cfg.mask_cleanup_ratio, max_size=41)
    target = _cleanup_mask(target, cleanup)

    if strong and np.any(target):
        dilate = _kernel_size(height, width, cfg.aggressive_dilate_ratio, max_size=61)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        target = cv2.dilate(target, kernel)

    return target, strong


def _sd_inpaint(
    image: np.ndarray,
    soil_mask: Optional[np.ndarray],
    snow_mask: Optional[np.ndarray],
    cfg: SnowInpaintConfig,
    sd_cfg: SDInpaintConfig,
    inpainter: SDInpainter,
) -> Optional[np.ndarray]:
    mask, strong = _prepare_sd_mask(image, soil_mask, snow_mask, cfg)
    if mask is None or not np.any(mask):
        return image
    height, width = image.shape[:2]
    target_size = _prepare_sd_size(width, height)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sd_image = _prepare_sd_image(image_rgb, target_size, Image.BICUBIC)

    mask_resized = mask
    if (width, height) != target_size:
        mask_resized = np.array(
            _prepare_sd_image(mask, target_size, Image.BILINEAR),
            dtype=np.uint8,
        )
    if sd_cfg.mask_blur_ratio > 0:
        mask_resized = _blur_mask(mask_resized, sd_cfg.mask_blur_ratio)
    sd_mask = Image.fromarray(mask_resized).convert("L")

    result = inpainter.inpaint(sd_image, sd_mask, strong)
    result_arr = np.array(result)
    if result_arr.shape[0] != height or result_arr.shape[1] != width:
        result_arr = np.array(result.resize((width, height), Image.BICUBIC))

    result_bgr = cv2.cvtColor(result_arr, cv2.COLOR_RGB2BGR)
    blend_mask = _blur_mask(mask, sd_cfg.mask_blur_ratio)
    return _blend_with_mask(image, result_bgr, blend_mask)


def _collect_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _resolve_output_path(input_root: Path, output_root: Path, source: Path) -> Path:
    relative = source.relative_to(input_root)
    dest = output_root / relative
    mapped = ENCODING_MAP.get(dest.suffix.lower())
    if mapped is None:
        return dest.with_suffix(".jpg")
    if mapped != dest.suffix.lower():
        return dest.with_suffix(mapped)
    return dest


def process_folder(
    input_dir: Path,
    output_dir: Path,
    cfg: SnowInpaintConfig,
    engine: str = "sd",
    sd_cfg: Optional[SDInpaintConfig] = None,
    overwrite: bool = False,
) -> Tuple[int, int]:
    processed = 0
    skipped = 0
    if engine not in {"sd", "opencv"}:
        raise ValueError(f"Unsupported engine: {engine}")
    inpainter = None
    if engine == "sd":
        if sd_cfg is None:
            sd_cfg = SDInpaintConfig()
        inpainter = SDInpainter(sd_cfg)

    for path in _collect_images(input_dir):
        output_path = _resolve_output_path(input_dir, output_dir, path)
        if output_path.exists() and not overwrite:
            skipped += 1
            continue

        image = _read_image(path)
        if image is None:
            skipped += 1
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        chroma = _compute_chroma(lab)
        ground_mask = _estimate_ground_mask(image, cfg)
        snow_mask = _detect_snow_mask(hsv, lab, chroma, cfg, ground_mask)
        soil_mask = _detect_soil_mask(hsv, lab, chroma, snow_mask, cfg, ground_mask)
        if engine == "sd":
            result = _sd_inpaint(image, soil_mask, snow_mask, cfg, sd_cfg, inpainter)
        else:
            result = _inpaint_snow(image, soil_mask, snow_mask, cfg)

        if not _write_image(output_path, result):
            skipped += 1
            continue
        processed += 1
    return processed, skipped


def _resolve_input_dir(arg_value: str) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    env_value = os.environ.get("SNOW_INPUT_DIR", "")
    if env_value:
        return Path(env_value).expanduser().resolve()
    raise ValueError("Input directory is required (use --input or SNOW_INPUT_DIR).")


def _resolve_output_dir(input_dir: Path, arg_value: str) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    env_value = os.environ.get("SNOW_OUTPUT_DIR", "")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return input_dir / "snow_inpainted"


def _resolve_engine(arg_value: str) -> str:
    if arg_value:
        return arg_value.strip().lower()
    env_value = os.environ.get("SNOW_ENGINE", "")
    if env_value:
        return env_value.strip().lower()
    return "sd"


def _resolve_float(arg_value: float, env_key: str, default: float) -> float:
    if arg_value:
        return float(arg_value)
    env_value = os.environ.get(env_key, "")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            return default
    return default


def _resolve_int(arg_value: int, env_key: str, default: int) -> int:
    if arg_value:
        return int(arg_value)
    env_value = os.environ.get(env_key, "")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            return default
    return default


def _resolve_optional_int(arg_value: int, env_key: str) -> Optional[int]:
    if arg_value:
        return int(arg_value)
    env_value = os.environ.get(env_key, "")
    if env_value:
        try:
            parsed = int(env_value)
        except ValueError:
            return None
        return parsed if parsed != 0 else None
    return None


def _resolve_bool(arg_value: bool, env_key: str, default: bool) -> bool:
    if arg_value:
        return True
    env_value = os.environ.get(env_key, "")
    if env_value:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _resolve_str(arg_value: str, env_key: str, default: str) -> str:
    if arg_value:
        return arg_value
    env_value = os.environ.get(env_key, "")
    if env_value:
        return env_value
    return default


def _resolve_sd_config(args: argparse.Namespace) -> SDInpaintConfig:
    base = SDInpaintConfig()
    attention_slicing = not args.sd_no_attention_slicing
    if not args.sd_no_attention_slicing:
        attention_slicing = _resolve_bool(False, "SD_ATTENTION_SLICING", base.attention_slicing)

    return SDInpaintConfig(
        model_id=_resolve_str(args.sd_model, "SD_MODEL", base.model_id),
        device=_resolve_str(args.sd_device, "SD_DEVICE", base.device),
        steps=_resolve_int(args.sd_steps, "SD_STEPS", base.steps),
        cfg_scale=_resolve_float(args.sd_cfg, "SD_CFG", base.cfg_scale),
        strength=_resolve_float(args.sd_strength, "SD_STRENGTH", base.strength),
        strength_strong=_resolve_float(args.sd_strength_strong, "SD_STRENGTH_STRONG", base.strength_strong),
        seed=_resolve_optional_int(args.sd_seed, "SD_SEED"),
        mask_blur_ratio=_resolve_float(args.sd_mask_blur_ratio, "SD_MASK_BLUR_RATIO", base.mask_blur_ratio),
        prompt=_resolve_str(args.sd_prompt, "SD_PROMPT", base.prompt),
        prompt_strong=_resolve_str(args.sd_prompt_strong, "SD_PROMPT_STRONG", base.prompt_strong),
        negative_prompt=_resolve_str(args.sd_negative, "SD_NEGATIVE_PROMPT", base.negative_prompt),
        xformers=_resolve_bool(args.sd_xformers, "SD_XFORMERS", base.xformers),
        attention_slicing=attention_slicing,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inpaint snow onto soil regions in photos.")
    parser.add_argument("--input", default="", help="Input folder with photos.")
    parser.add_argument("--output", default="", help="Output folder (default: input/snow_inpainted).")
    parser.add_argument("--engine", default="", help="Inpaint engine: sd or opencv (default: sd).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    parser.add_argument("--sd-model", default="", help="SD inpaint model (HF repo id or local path).")
    parser.add_argument("--sd-device", "--device", dest="sd_device", default="", help="Device: auto, cuda, cpu.")
    parser.add_argument("--sd-steps", "--steps", dest="sd_steps", type=int, default=0, help="SD steps.")
    parser.add_argument("--sd-cfg", "--cfg", dest="sd_cfg", type=float, default=0.0, help="SD CFG scale.")
    parser.add_argument("--sd-strength", "--strength", dest="sd_strength", type=float, default=0.0, help="SD denoising strength.")
    parser.add_argument(
        "--sd-strength-strong",
        type=float,
        default=0.0,
        help="SD denoising strength when snow already exists.",
    )
    parser.add_argument("--sd-seed", "--seed", dest="sd_seed", type=int, default=0, help="SD seed (0 = random).")
    parser.add_argument("--sd-mask-blur-ratio", type=float, default=0.0, help="SD mask blur ratio.")
    parser.add_argument("--sd-prompt", "--prompt", dest="sd_prompt", default="", help="SD prompt.")
    parser.add_argument("--sd-prompt-strong", default="", help="SD prompt for stronger snow.")
    parser.add_argument("--sd-negative", "--negative-prompt", dest="sd_negative", default="", help="SD negative prompt.")
    parser.add_argument("--sd-xformers", action="store_true", help="Enable xformers attention (if installed).")
    parser.add_argument(
        "--sd-no-attention-slicing",
        action="store_true",
        help="Disable attention slicing (uses more VRAM).",
    )
    args = parser.parse_args()

    try:
        input_dir = _resolve_input_dir(args.input)
    except ValueError as exc:
        print(str(exc))
        return 2

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 2

    output_dir = _resolve_output_dir(input_dir, args.output)
    cfg = SnowInpaintConfig()
    engine = _resolve_engine(args.engine)
    sd_cfg = _resolve_sd_config(args) if engine == "sd" else None

    try:
        processed, skipped = process_folder(
            input_dir,
            output_dir,
            cfg,
            engine=engine,
            sd_cfg=sd_cfg,
            overwrite=args.overwrite,
        )
    except (ValueError, RuntimeError, SystemExit) as exc:
        print(str(exc))
        return 2
    print(f"Processed {processed} images. Skipped {skipped} items.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
