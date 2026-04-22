# model.py
import numpy as np
import cv2
import json
from pathlib import Path

def load_images(folder):
    images = []
    for p in Path(folder).glob("*.jpg"):
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        images.append(img)
    return images


def compute_luminance(img):
    # Rec. 709
    return 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]


def fit_tone_curve(images, n_bins=256):
    all_lums = []

    for img in images:
        lum = compute_luminance(img)
        all_lums.append(lum.flatten())

    all_lums = np.concatenate(all_lums)

    # Histogram → CDF
    hist, bins = np.histogram(all_lums, bins=n_bins, range=(0,1), density=True)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]

    return {
        "bins": bins.tolist(),
        "cdf": cdf.tolist()
    }


def apply_tone_curve(img, curve):
    bins = np.array(curve["bins"])
    cdf = np.array(curve["cdf"])

    lum = compute_luminance(img)

    mapped = np.interp(lum.flatten(), bins[:-1], cdf).reshape(lum.shape)

    # scale RGB by luminance ratio
    eps = 1e-6
    ratio = mapped / (lum + eps)
    out = img * ratio[:,:,None]

    return np.clip(out, 0, 1)


def fit_color_matrix(images):
    # naive: match mean color to neutral gray
    mean_colors = []

    for img in images:
        mean_colors.append(img.reshape(-1,3).mean(axis=0))

    mean_color = np.mean(mean_colors, axis=0)

    # scale each channel to equalize
    target = np.mean(mean_color)
    scale = target / (mean_color + 1e-6)

    matrix = np.diag(scale)

    return matrix.tolist()


def apply_color_matrix(img, matrix):
    M = np.array(matrix)
    h, w, _ = img.shape
    reshaped = img.reshape(-1,3)
    transformed = reshaped @ M.T
    return np.clip(transformed.reshape(h,w,3), 0, 1)


def save_profile(profile, path):
    with open(path, "w") as f:
        json.dump(profile, f)


def load_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def train_profile(image_folder, output_path):
    images = load_images(image_folder)

    tone_curve = fit_tone_curve(images)
    color_matrix = fit_color_matrix(images)

    profile = {
        "tone_curve": tone_curve,
        "color_matrix": color_matrix
    }

    save_profile(profile, output_path)
    return profile


def apply_profile(img, profile):
    img = apply_tone_curve(img, profile["tone_curve"])
    img = apply_color_matrix(img, profile["color_matrix"])
    return img