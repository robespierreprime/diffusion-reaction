import os
import numpy
import time
import scipy.signal
from PIL import Image

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

GRID_SIZE_REF = 128
DT_REF = 1.0
N_STEPS_REF = 2400

GRID_SIZE = 512

FEED_RATE = 0.0545
KILL_RATE = 0.062

DIFUSION_A = 0.2
DIFUSION_B = 0.1

OUTPUT_DIR = "images"


scale = (GRID_SIZE / GRID_SIZE_REF) ** 2
DT = DT_REF / scale
N_STEPS = int(N_STEPS_REF * scale)
SAVE_EVERY = max(1, N_STEPS // 720)  # ~720 (24*30s)

kernel = numpy.array(
    [
        [0.05, 0.2, 0.05],
        [0.2, -1, 0.2],
        [0.05, 0.2, 0.05],
    ],
    dtype="float32",
)


def initialize_b(shape, p=0.05):
    return (numpy.random.random(shape) < p).astype("float32")


def update(A, B):
    AB2 = A * B * B

    DA = scipy.signal.fftconvolve(A, kernel, mode="same")
    DB = scipy.signal.fftconvolve(B, kernel, mode="same")

    A += DT * (DIFUSION_A * scale * DA - AB2 + FEED_RATE * (1 - A))
    B += DT * (DIFUSION_B * scale * DB + AB2 - (KILL_RATE + FEED_RATE) * B)
    numpy.clip(A, 0.0, 1.0, out=A)
    numpy.clip(B, 0.0, 1.0, out=B)
    return A, B


def main():
    print(f"GRID_SIZE={GRID_SIZE}  scale={scale:.3f}  DT={DT:.6f}  N_STEPS={N_STEPS}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    A = numpy.ones((GRID_SIZE, GRID_SIZE), dtype="float32")
    B = initialize_b((GRID_SIZE, GRID_SIZE))

    frame = 0
    for step in range(N_STEPS):
        A, B = update(A, B)

        if step % SAVE_EVERY == 0:
            img = Image.fromarray((B * 255).astype("uint8"))
            img.save(os.path.join(OUTPUT_DIR, f"{frame:05d}.png"))
            frame += 1
            print(step)


if __name__ == "__main__":
    main()
