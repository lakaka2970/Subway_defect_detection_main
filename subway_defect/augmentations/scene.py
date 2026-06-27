"""
Scene-specific augmentations for subway catenary imagery.

Simulates six lighting/environmental conditions:
  - tunnelize:        dark tunnel lighting (yellow sodium lamps + low exposure)
  - sunlitize:         bright outdoor sunlight (high contrast + shadows)
  - motion_blur:       directional vehicle-motion blur (long-exposure smear)
  - weather_augment:   rain streaks + fog overlay
  - vibration_blur:    high-frequency micro-vibration (Gaussian blur + bidirectional pixel shift)
  - white_balance_shift: camera white-balance / colour-temperature drift (4 lighting modes)

All functions accept and return np.ndarray (H, W, 3) uint8 BGR.
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


def vibration_blur(img: np.ndarray) -> np.ndarray:
    """Simulate high-frequency micro-vibration from passing trains.

    Unlike ``motion_blur`` (directional, long-distance smear), this produces
    localised high-frequency jitter by applying Gaussian blur followed by a
    subtle bidirectional pixel shift — the visual signature of rail/train
    vibration transmitted through the catenary structure.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        Vibration-blurred image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Gaussian blur (high-frequency vibration)
    sigma = np.random.uniform(1.0, 2.5)
    kernel_size = np.random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # Subtle bidirectional pixel shift (simulate structural resonance)
    shift_x = np.random.randint(-2, 3)
    shift_y = np.random.randint(-2, 3)
    if shift_x != 0 or shift_y != 0:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def white_balance_shift(img: np.ndarray) -> np.ndarray:
    """Simulate different tunnel lighting colour temperatures.

    Applies independent per-channel gain to mimic the transition from warm
    sodium-vapour lamps (yellow/orange cast) to cool LED panels (blue cast)
    or fluorescent strips (green cast). Also models partial AWB failure
    where one channel drifts to an extreme.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        White-balance-shifted image, same shape and dtype.
    """
    img = img.copy().astype(np.float32)

    # BGR channel gains
    mode = np.random.choice(["warm", "cool", "fluorescent", "awb_fail"],
                            p=[0.40, 0.30, 0.20, 0.10])

    if mode == "warm":
        # Sodium-vapour: boost red, suppress blue (BGR: B↓, G→, R↑)
        gains = np.array([
            np.random.uniform(0.65, 0.90),  # B: suppressed
            np.random.uniform(0.90, 1.05),  # G: neutral
            np.random.uniform(1.10, 1.40),  # R: boosted
        ], dtype=np.float32).reshape(1, 1, 3)
    elif mode == "cool":
        # LED: boost blue, suppress red (BGR: B↑, G→, R↓)
        gains = np.array([
            np.random.uniform(1.10, 1.45),  # B: boosted
            np.random.uniform(0.90, 1.05),  # G: neutral
            np.random.uniform(0.65, 0.90),  # R: suppressed
        ], dtype=np.float32).reshape(1, 1, 3)
    elif mode == "fluorescent":
        # Green shift: boost green (BGR: B→, G↑, R→)
        gains = np.array([
            np.random.uniform(0.85, 1.05),  # B: near neutral
            np.random.uniform(1.10, 1.35),  # G: boosted
            np.random.uniform(0.85, 1.05),  # R: near neutral
        ], dtype=np.float32).reshape(1, 1, 3)
    else:  # awb_fail
        # Single-channel extreme drift
        channel = np.random.randint(0, 3)
        gains = np.ones(3, dtype=np.float32)
        gains[channel] = np.random.uniform(1.30, 1.60)
        gains = gains.reshape(1, 1, 3)

    # Apply gains and add subtle colour noise
    img = img * gains
    noise = np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0, 3)
    img = (img + noise).clip(0, 255).astype(np.uint8)

    return img
