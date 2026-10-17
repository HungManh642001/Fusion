"""SAR Backscatter Radiometric Calibration and 2D CA-CFAR Ship Detector.

Implements Sentinel-1 / xView3 SAR processing:
1. Radiometric calibration from raw DN to Sigma0 (dB)
2. Land masking
3. 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) ship detection
4. Connected-component vessel dimension estimation (Length, Breadth)
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy.ndimage import uniform_filter, label, find_objects


class SARPreprocessor:
    """Processes satellite SAR imagery for maritime vessel reconnaissance."""

    def __init__(self, pixel_resolution_m: float = 10.0):
        self.pixel_resolution_m = pixel_resolution_m

    def calibrate_sigma0(self, raw_dn: np.ndarray, ks: float = 30.0) -> np.ndarray:
        """Calibrate digital numbers (DN) to normalized radar cross section sigma_0 (dB)."""
        dn_safe = np.maximum(raw_dn.astype(float), 1e-6)
        sigma0_linear = dn_safe**2
        sigma0_db = 10.0 * np.log10(sigma0_linear) - float(ks)
        return sigma0_db

    def apply_land_mask(self, image: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
        """Zero-out land pixels so detectors focus exclusively on sea surfaces."""
        masked = image.copy()
        masked[land_mask > 0] = -999.0
        return masked

    def extract_cfar_targets(
        self,
        sigma0_img: np.ndarray,
        guard_size: int = 3,
        train_size: int = 7,
        threshold_factor: float = 2.5,
    ) -> List[Dict[str, Any]]:
        """Extract vessel target candidate centroids using 2D CA-CFAR filter.

        Args:
            sigma0_img: 2D array of backscatter intensity
            guard_size: Half-width of guard cell window
            train_size: Half-width of reference/training window
            threshold_factor: Multiplier for local standard deviation

        Returns:
            List of detected target dicts with pixel coordinates and estimated lengths
        """
        img = sigma0_img.astype(float)
        # Linear scale for CFAR statistics
        img_lin = np.power(10.0, np.clip(img, -50.0, 50.0) / 10.0)

        w_train = 2 * train_size + 1
        w_guard = 2 * guard_size + 1

        # Use fast 2D uniform filter for mean in windows
        train_mean = uniform_filter(img_lin, size=w_train, mode="reflect")
        guard_mean = uniform_filter(img_lin, size=w_guard, mode="reflect")

        area_train = w_train * w_train
        area_guard = w_guard * w_guard
        area_ref = area_train - area_guard

        # Background clutter mean excluding guard cells
        clutter_mean = (train_mean * area_train - guard_mean * area_guard) / max(area_ref, 1)
        clutter_mean = np.maximum(clutter_mean, 1e-12)

        # Local standard deviation approximation
        train_sq_mean = uniform_filter(img_lin**2, size=w_train, mode="reflect")
        guard_sq_mean = uniform_filter(img_lin**2, size=w_guard, mode="reflect")
        clutter_sq_mean = (train_sq_mean * area_train - guard_sq_mean * area_guard) / max(area_ref, 1)
        clutter_var = np.maximum(clutter_sq_mean - clutter_mean**2, 1e-12)
        clutter_std = np.sqrt(clutter_var)

        # CFAR threshold map
        cfar_threshold = clutter_mean + threshold_factor * clutter_std
        detection_mask = (img_lin > cfar_threshold) & (img > -30.0)

        # Group adjacent detection pixels into targets
        labeled_mask, num_features = label(detection_mask)
        objects = find_objects(labeled_mask)

        detections = []
        for i, sl in enumerate(objects):
            if sl is None:
                continue
            r_slice, c_slice = sl
            patch_mask = labeled_mask[r_slice, c_slice] == (i + 1)
            num_pixels = int(np.sum(patch_mask))
            if num_pixels < 1:
                continue

            r_indices, c_indices = np.where(patch_mask)
            center_r = float(r_slice.start + np.mean(r_indices))
            center_c = float(c_slice.start + np.mean(c_indices))

            # Length estimate from bounding box diagonal scaled by resolution
            box_height_m = (r_slice.stop - r_slice.start) * self.pixel_resolution_m
            box_width_m = (c_slice.stop - c_slice.start) * self.pixel_resolution_m
            estimated_length_m = float(np.hypot(box_height_m, box_width_m))

            max_db = float(np.max(img[r_slice, c_slice][patch_mask]))

            detections.append({
                "detection_id": i + 1,
                "pixel_r": center_r,
                "pixel_c": center_c,
                "length_m": max(estimated_length_m, 15.0),
                "num_pixels": num_pixels,
                "peak_sigma0_db": max_db,
            })

        return detections
