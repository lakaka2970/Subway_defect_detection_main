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


def glare_augment(img: np.ndarray, p_streak: float = 0.5) -> np.ndarray:
    """Simulate reflective glare from metal surfaces and catenary wires.

    Models two real-world scenarios in subway tunnels:
    1. Specular highlights on polished metal parts (localized bright blobs)
    2. Light streaks from overhead wires / rails (elongated saturated bands)

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_streak: Probability of adding directional light streaks.

    Returns:
        Glare-augmented image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # Localized specular highlights (1-4 bright blobs)
    n_blobs = np.random.randint(1, 5)
    for _ in range(n_blobs):
        cx = np.random.randint(w // 8, 7 * w // 8)
        cy = np.random.randint(h // 8, 7 * h // 8)
        radius = np.random.randint(max(8, w // 40), max(16, w // 10))
        intensity = np.random.uniform(0.4, 0.9)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = np.exp(-(dist ** 2) / (2 * (radius / 2.0) ** 2))
        blob = np.clip(blob * intensity, 0, 1)

        # Slightly warm or cool tint for the glare
        tint = np.array([
            np.random.uniform(0.9, 1.0),
            np.random.uniform(0.9, 1.0),
            np.random.uniform(0.95, 1.0),
        ], dtype=np.float32).reshape(1, 1, 3)
        out = out * (1 - blob[..., None]) + 255.0 * blob[..., None] * tint

    # Directional light streaks (overhead wire reflections)
    if np.random.random() < p_streak:
        n_streaks = np.random.randint(1, 4)
        for _ in range(n_streaks):
            angle = np.random.uniform(-30, 30)  # near-horizontal
            thickness = np.random.randint(2, max(3, w // 100))
            y_pos = np.random.randint(h // 6, 5 * h // 6)
            length = np.random.randint(w // 3, w)
            x_start = np.random.randint(0, max(1, w - length))
            streak_intensity = np.random.uniform(0.3, 0.7)

            # Draw a soft streak using a rotated rectangle mask
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.line(mask, (x_start, y_pos),
                     (x_start + length, y_pos + int(length * np.tan(np.radians(angle)))),
                     1.0, thickness=thickness, lineType=cv2.LINE_AA)
            # Gaussian blur for soft edges
            ksize = max(3, thickness * 2 + 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
            mask = mask * streak_intensity

            out = out * (1 - mask[..., None]) + 255.0 * mask[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def night_augment(img: np.ndarray, p_ir: float = 0.3) -> np.ndarray:
    """Simulate night / low-light inspection conditions.

    Models two scenarios:
    1. Visible-light camera in very low ambient light (severe brightness drop,
       high sensor noise, slight blue cast from moonlight / LED floodlights)
    2. Near-infrared (IR) camera mode (desaturated, high contrast, grainy)

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_ir: Probability of simulating IR mode instead of visible low-light.

    Returns:
        Night-augmented image, same shape and dtype.
    """
    img = img.copy()
    h, w = img.shape[:2]

    if np.random.random() < p_ir:
        # IR camera simulation: desaturate + contrast stretch + grain
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Contrast stretch (histogram equalization-like effect)
        p_low, p_high = np.percentile(gray, [2, 98])
        if p_high - p_low > 10:
            gray = (gray - p_low) / (p_high - p_low) * 255.0
        gray = np.clip(gray, 0, 255)

        # Slight blue-green tint typical of IR sensors
        out = np.stack([
            gray * np.random.uniform(0.85, 0.95),  # B
            gray * np.random.uniform(0.95, 1.05),  # G
            gray * np.random.uniform(0.80, 0.90),  # R
        ], axis=-1)

        # Heavy sensor noise (IR sensors are noisier)
        noise_sigma = np.random.uniform(8, 20)
        noise = np.random.randn(*out.shape).astype(np.float32) * noise_sigma
        out = out + noise

    else:
        # Visible low-light: severe brightness drop + blue cast + noise
        brightness = np.random.uniform(0.10, 0.30)
        out = img.astype(np.float32) * brightness

        # Blue-ish cast from LED floodlights / moonlight
        blue_cast = np.array([
            np.random.uniform(1.05, 1.25),  # B boost
            np.random.uniform(0.95, 1.05),  # G
            np.random.uniform(0.80, 0.95),  # R suppress
        ], dtype=np.float32).reshape(1, 1, 3)
        out = out * blue_cast

        # Heavy sensor noise in low light (shot noise + read noise)
        noise_sigma = np.random.uniform(10, 25)
        noise = np.random.randn(*out.shape).astype(np.float32) * noise_sigma
        out = out + noise

        # Occasional vignetting from lens hood
        if np.random.random() < 0.4:
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            vignette = 1.0 - 0.4 * (dist / max_dist) ** 2
            out = out * vignette[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)
