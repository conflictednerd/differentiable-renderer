# A minimal, fully-differentiable 2D renderer for circles, implemented in JAX.

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(1, 2, 3))
def render(circles, width, height, softness=10.0):
    """
    Renders an entire image by alpha-compositing circles.

    Args:
        circles (jnp.ndarray): A (N, 7) array of circle parameters.
        width (int): The width of the canvas.
        height (int): The height of the canvas.
        softness (float): Controls the anti-aliasing of circle edges.

    Returns:
        jnp.ndarray: An (H, W, 3) array representing the rendered image.
    """
    num_circles = circles.shape[0]

    y_coords, x_coords = jnp.meshgrid(
        jnp.linspace(0, float(height - 1), height),
        jnp.linspace(0, float(width - 1), width),
        indexing="ij",
    )
    x_coords = x_coords[:, :, jnp.newaxis]  # (H, W, 1)
    y_coords = y_coords[:, :, jnp.newaxis]  # (H, W, 1)

    # Initialize a white canvas
    canvas = jnp.full((height, width, 3), 1.0)

    cx_logit = circles[:, 0]
    cy_logit = circles[:, 1]
    radius_logit = circles[:, 2]
    r_logit = circles[:, 3]
    g_logit = circles[:, 4]
    b_logit = circles[:, 5]
    alpha_logit = circles[:, 6]

    cx = jax.nn.sigmoid(cx_logit) * float(width)
    cy = jax.nn.sigmoid(cy_logit) * float(height)
    max_radius = float(min(width, height)) / 2.0
    radius = jax.nn.sigmoid(radius_logit) * max_radius + 1.0
    color_circle_logits = jnp.stack([r_logit, g_logit, b_logit], axis=1)
    color_circles = jax.nn.sigmoid(color_circle_logits) * 2.0 - 1.0
    alpha_circles = jax.nn.sigmoid(alpha_logit)

    # Reshape parameters for broadcasting against the coordinate grid
    cx = cx.reshape(1, 1, num_circles)
    cy = cy.reshape(1, 1, num_circles)
    radius = radius.reshape(1, 1, num_circles)
    color_circles = color_circles.reshape(1, 1, num_circles, 3)
    alpha_circles = alpha_circles.reshape(1, 1, num_circles)

    # Compute distances for all circles at once
    # Broadcasting (H, W, 1) with (1, 1, N) results in (H, W, N)
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    val = (dist_sq - radius**2) / softness

    # Calculate alpha values for every pixel for every circle.
    pixel_alphas = alpha_circles * (1.0 - jax.nn.sigmoid(val))  # (H, W, N)
    pixel_alphas_reshaped = pixel_alphas[:, :, :, jnp.newaxis]  # (H, W, N, 1)

    pixel_colors = color_circles * pixel_alphas_reshaped  # (H, W, N, 3)
    opacities = 1.0 - pixel_alphas_reshaped  # (H, W, N, 1)
    transparencies = jnp.concatenate(
        [jnp.full((height, width, 1, 1), 1.0), opacities], axis=2
    )[
        :, :, :-1, :
    ]  # (H, W, N, 1)

    transparency_matrix = jnp.cumprod(transparencies, axis=2)  # (H, W, N, 1)

    foreground_color = jnp.sum(transparency_matrix * pixel_colors, axis=2)
    background_contribution = canvas * jnp.prod(opacities, axis=2)
    final_canvas = foreground_color + background_contribution

    return final_canvas


@partial(jit, static_argnums=(2, 3))
def mse_loss(circles, target_image, width, height, softness=10.0):
    """
    Calculates the Mean Squared Error.
    """
    rendered_image = render(circles, width, height, softness)
    return jnp.mean((rendered_image - target_image) ** 2)
