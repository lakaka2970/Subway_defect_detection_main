"""
Scene-specific augmentations for subway catenary imagery.

Simulates: tunnel lighting (dark + yellow spotlights), outdoor sunlight
(high contrast + shadows), motion blur (vehicle vibration), and weather
(fog, rain). All functions accept and return np.ndarray (H, W, 3) uint8 BGR.
"""

import cv2
import numpy as np


def tunnelize(img: np.ndarray, p_brightness: float = 0.5) -> np.ndarray:
    """Simulate tunnel lighting: dark, yellow spotlight, sensor noise.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_brightness: Probability of adding a warm spotlight.

    Returns:
        Augmented image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Global brightness reduction
    brightness = np.random.uniform(0.3, 0.6)
    img = (img.astype(np.float32) * brightness).clip(0, 255)

    # Warm spotlight from train headlights
    if np.random.random() < p_brightness:
        cy = np.random.randint(h // 4, 3 * h // 4)
        cx = w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + ((y - cy) * 2.5) ** 2)
        spotlight = np.exp(-r / (w * 0.12))
        spotlight = np.clip(spotlight * 1.8, 0, 1)
        warm = np.array([120, 180, 255], dtype=np.float32).reshape(1, 1, 3)
        img = (img.astype(np.float32) * (1 + spotlight[..., None] * 0.6)
               + spotlight[..., None] * warm * 0.5)

    # Sensor noise in low light
    noise_sigma = np.random.uniform(3, 10)
    noise = np.random.randn(*img.shape).astype(np.float32) * noise_sigma
    img = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    return img


def sunlitize(img: np.ndarray, p_shadow: float = 0.4) -> np.ndarray:
    """Simulate strong outdoor sunlight: brightness boost + gradient shadows.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_shadow: Probability of adding structural shadow strips.

    Returns:
        Augmented image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Global brightness boost
    scale = np.random.uniform(1.2, 1.7)
    img = (img.astype(np.float32) * scale).clip(0, 255)

    # Gradient shadow strips
    if np.random.random() < p_shadow:
        shadow = np.ones((h, w), dtype=np.float32)
        n_strips = np.random.randint(1, 5)
        for _ in range(n_strips):
            x0 = np.random.randint(0, w)
            direction = np.sign(np.random.randn())
            grad = np.tile(
                np.linspace(0.4, 1.0, np.random.randint(w // 6, w // 2)),
                (h, 1),
            )
            if direction < 0:
                grad = np.fliplr(grad)
            x1 = min(x0 + grad.shape[1], w)
            shadow[:, x0:x1] = np.minimum(shadow[:, x0:x1],
                                          grad[:, :x1 - x0])
        img = (img.astype(np.float32) * shadow[..., None]).clip(0, 255)

    return img.astype(np.uint8)


def motion_blur(img: np.ndarray) -> np.ndarray:
    """Simulate vehicle vibration blur with random kernel length and angle.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        Motion-blurred image, same shape and dtype.
    """
    img = img.copy()
    length = np.random.randint(3, 10)
    angle = np.random.uniform(0, 360)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))

    size = max(1, length)
    kernel = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    for i in range(length):
        x = int(cx + (i - length / 2) * cos_a)
        y = int(cy + (i - length / 2) * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def vibration_blur(img: np.ndarray) -> np.ndarray:
    """Alias for high-frequency vehicle vibration blur."""
    return motion_blur(img)


def white_balance_shift(img: np.ndarray) -> np.ndarray:
    """Simulate tunnel lighting color-temperature shifts."""
    img = img.copy().astype(np.float32)
    gains = np.array([
        np.random.uniform(0.85, 1.15),
        np.random.uniform(0.90, 1.10),
        np.random.uniform(0.85, 1.20),
    ], dtype=np.float32).reshape(1, 1, 3)
    return (img * gains).clip(0, 255).astype(np.uint8)


def weather_augment(img: np.ndarray) -> np.ndarray:
    """Apply random weather effect: fog or rain.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        Weather-augmented image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]

    if np.random.random() < 0.6:
        # Fog: exponential-decay white overlay
        intensity = np.random.uniform(0.15, 0.45)
        fog_color = np.random.randint(200, 255, 3, dtype=np.uint8)
        fog_color = fog_color.astype(np.float32).reshape(1, 1, 3)
        y, x = np.ogrid[:h, :w]
        cy = np.random.randint(h // 4, 3 * h // 4)
        cx = np.random.randint(w // 4, 3 * w // 4)
        dist = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.8) ** 2)
        mask = np.exp(-dist / (w * 0.2)) * intensity
        mask = np.clip(mask, 0, 1)[..., None]
        img = (img.astype(np.float32) * (1 - mask)
               + fog_color * mask).clip(0, 255).astype(np.uint8)
    else:
        # Rain: sparse short lines
        n_drops = np.random.randint(15, 60)
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(3, 12)
            angle = np.random.uniform(70, 110)
            dx = int(length * np.cos(np.radians(angle)))
            dy = int(length * np.sin(np.radians(angle)))
            cv2.line(img, (x, y), (x + dx, y + dy), (200, 210, 220),
                     thickness=1, lineType=cv2.LINE_AA)
    return img
