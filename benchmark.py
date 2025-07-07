# A utility to compare the performance of the JAX and PyTorch differentiable renderers.
# This version normalizes images to [-1, 1], uses uniform parameter initialization,
# and saves a visual comparison of the optimization process.

import time
import argparse
from pathlib import Path
import numpy as onp
from PIL import Image, ImageDraw
from tqdm import trange

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import optax
from renderer_jax import render as render_jax, mse_loss as mse_loss_jax

import torch
from torch import optim
from renderer_torch import render as render_pytorch, mse_loss as mse_loss_pytorch


# Configuration
IMG_SIZE = 256
NUM_CIRCLES = 2048
LEARNING_RATE = 0.05
STEPS = 4096


# Helper Functions
def denormalize_image(img_np):
    """Converts an image from [-1, 1] float to [0, 255] uint8."""
    return onp.clip((img_np + 1.0) * 127.5, 0, 255).astype(onp.uint8)


def preprocess_image(image_path):
    """Loads, resizes, and normalizes an image to [-1, 1]."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        # Normalize to [-1, 1]
        img_np = (onp.array(img, dtype=onp.float32) / 127.5) - 1.0
        return img_np
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None


def save_comparison_image(
    initial_img_np, target_img_np, final_img_np, title, output_path
):
    """Saves a single image comparing the initial, target, and final states."""
    # Convert float arrays to PIL Images
    initial_pil = Image.fromarray(denormalize_image(initial_img_np))
    target_pil = Image.fromarray(denormalize_image(target_img_np))
    final_pil = Image.fromarray(denormalize_image(final_img_np))

    # Create a new composite image
    width, height = initial_pil.size
    padding = 20
    text_height = 40
    total_width = width * 3 + padding * 2
    total_height = height + text_height + padding

    composite_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(composite_img)

    # Paste images
    composite_img.paste(initial_pil, (0, 0))
    composite_img.paste(target_pil, (width + padding, 0))
    composite_img.paste(final_pil, (width * 2 + padding * 2, 0))

    # Add titles
    draw.text((width / 2, height + 5), "Initial", fill="black", anchor="mt")
    draw.text(
        (width + padding + width / 2, height + 5), "Target", fill="black", anchor="mt"
    )
    draw.text(
        (width * 2 + padding * 2 + width / 2, height + 5),
        "Final",
        fill="black",
        anchor="mt",
    )
    draw.text((total_width / 2, height + 25), title, fill="black", anchor="mt")

    composite_img.save(output_path)
    print(f"Comparison image saved to {output_path}")


# Optimizations
def run_jax_optimization(target_image_np, output_path):
    """Runs a full optimization loop using the JAX renderer."""
    print(f"\n--- Running JAX Benchmark ---")
    target_image_jax = jnp.array(target_image_np)

    # Use uniform initialization in range [-2, 2]
    key = jax.random.PRNGKey(0)
    initial_params = jax.random.uniform(key, (NUM_CIRCLES, 7), minval=-2.0, maxval=2.0)
    circles_params = initial_params

    # Render initial state for comparison
    initial_render = render_jax(initial_params, IMG_SIZE, IMG_SIZE)

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(circles_params)
    loss_fn_for_grad = partial(mse_loss_jax, width=IMG_SIZE, height=IMG_SIZE)
    loss_grad_fn = jit(grad(loss_fn_for_grad))

    start_time = time.time()
    for _ in trange(STEPS, desc="JAX Steps"):
        grads = loss_grad_fn(circles_params, target_image_jax)
        updates, opt_state = optimizer.update(grads, opt_state)
        circles_params = optax.apply_updates(circles_params, updates)

    jax.block_until_ready(circles_params)
    end_time = time.time()

    final_render = render_jax(circles_params, IMG_SIZE, IMG_SIZE)
    final_loss = mse_loss_jax(circles_params, target_image_jax, IMG_SIZE, IMG_SIZE)

    save_comparison_image(
        onp.array(initial_render),
        target_image_np,
        onp.array(final_render),
        "Optimization with JAX Renderer",
        output_path,
    )
    return end_time - start_time, float(final_loss)


def run_pytorch_optimization(target_image_np, output_path):
    """Runs the full optimization loop using the PyTorch renderer."""
    print(f"\n--- Running PyTorch Benchmark ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using PyTorch device: {device}")
    target_image_torch = torch.from_numpy(target_image_np).to(device)

    # Use uniform initialization in range [-2, 2]
    torch.manual_seed(0)
    initial_params = torch.rand(NUM_CIRCLES, 7, device=device) * 4.0 - 2.0
    initial_params.requires_grad_()
    circles_params = initial_params

    # Render initial state for comparison
    initial_render = render_pytorch(initial_params.detach(), IMG_SIZE, IMG_SIZE)

    optimizer = optim.Adam([circles_params], lr=LEARNING_RATE)

    start_time = time.time()
    for _ in trange(STEPS, desc="PyTorch Steps"):
        optimizer.zero_grad()
        rendered_image = render_pytorch(circles_params, IMG_SIZE, IMG_SIZE)
        loss = mse_loss_pytorch(rendered_image, target_image_torch)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    final_render = render_pytorch(circles_params.detach(), IMG_SIZE, IMG_SIZE)
    final_loss = mse_loss_pytorch(final_render, target_image_torch)

    save_comparison_image(
        initial_render.detach().cpu().numpy(),
        target_image_np,
        final_render.detach().cpu().numpy(),
        "Optimization with PyTorch Renderer",
        output_path,
    )
    return end_time - start_time, float(final_loss.item())


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX and PyTorch renderers.")
    parser.add_argument("image_files", nargs="+", help="Paths to images.")
    args = parser.parse_args()

    results = []
    output_dir = Path("./benchmark_outputs")
    output_dir.mkdir(exist_ok=True)

    for image_file in args.image_files:
        print(
            f"\n=========================================\nProcessing: {image_file}\n========================================="
        )
        target_image_np = preprocess_image(image_file)
        if target_image_np is None:
            continue

        base_name = Path(image_file).stem

        jax_time, jax_loss = run_jax_optimization(
            target_image_np, output_dir / f"{base_name}_comparison_jax.png"
        )
        pytorch_time, pytorch_loss = run_pytorch_optimization(
            target_image_np, output_dir / f"{base_name}_comparison_pytorch.png"
        )

        results.append(
            {
                "image": base_name,
                "jax_time": jax_time,
                "jax_loss": jax_loss,
                "pytorch_time": pytorch_time,
                "pytorch_loss": pytorch_loss,
            }
        )

    print(
        "\n\n=========================================\n           Benchmark Results\n========================================="
    )
    print(f"{'Image':<20} | {'Framework':<10} | {'Time (s)':<12} | {'Final Loss':<15}")
    print("-" * 65)
    for res in results:
        print(
            f"{res['image']:<20} | {'JAX':<10} | {res['jax_time']:<12.4f} | {res['jax_loss']:<15.6f}"
        )
        print(
            f"{res['image']:<20} | {'PyTorch':<10} | {res['pytorch_time']:<12.4f} | {res['pytorch_loss']:<15.6f}"
        )
        print("-" * 65)


if __name__ == "__main__":
    main()
