# A minimal, fully-differentiable 2D renderer for circles, implemented in PyTorch.

import torch


@torch.jit.script
def render(circles, width: int, height: int, softness: float = 10.0):
    """
    Renders an entire image by alpha-compositing circles onto a grid.
    This method will run on whatever device `circles` is on.

    Args:
        circles (torch.Tensor): A (N, 7) tensor of circle parameters.
        width (int): The width of the canvas.
        height (int): The height of the canvas.
        softness (float): Controls the anti-aliasing of circle edges.

    Returns:
        torch.Tensor: An (H, W, 3) tensor representing the rendered image.
    """
    device = circles.device
    num_circles = circles.shape[0]

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0, float(height - 1), height, device=device),
        torch.linspace(0, float(width - 1), width, device=device),
        indexing="ij",
    )
    x_coords = x_coords.unsqueeze(2)  # (H, W, 1)
    y_coords = y_coords.unsqueeze(2)  # (H, W, 1)

    canvas = torch.full((height, width, 3), 1.0, device=device)

    cx_logit = circles[:, 0]
    cy_logit = circles[:, 1]
    radius_logit = circles[:, 2]
    r_logit = circles[:, 3]
    g_logit = circles[:, 4]
    b_logit = circles[:, 5]
    alpha_logit = circles[:, 6]

    cx = torch.sigmoid(cx_logit) * float(width)
    cy = torch.sigmoid(cy_logit) * float(height)
    max_radius = float(min(width, height)) / 2.0
    radius = torch.sigmoid(radius_logit) * max_radius + 1.0
    color_circle_logits = torch.stack([r_logit, g_logit, b_logit], dim=1)
    color_circles = torch.sigmoid(color_circle_logits) * 2.0 - 1.0
    alpha_circles = torch.sigmoid(alpha_logit)

    # Reshape for broadcasting against the coordinate grid
    cx = cx.view(1, 1, num_circles)
    cy = cy.view(1, 1, num_circles)
    radius = radius.view(1, 1, num_circles)
    color_circles = color_circles.view(1, 1, num_circles, 3)
    alpha_circles = alpha_circles.view(1, 1, num_circles)

    # Compute distances for all circles at once
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    val = (dist_sq - radius**2) / softness

    pixel_alphas = alpha_circles * (1.0 - torch.sigmoid(val))  # (H, W, N)
    pixel_alphas_reshaped = pixel_alphas.unsqueeze(3)  # (H, W, N, 1)

    pixel_colors = color_circles * pixel_alphas_reshaped

    # Alpha-composite all circle layers
    opacities = 1.0 - pixel_alphas_reshaped
    transparencies = torch.cat(
        [torch.full((height, width, 1, 1), 1.0, device=device), opacities], dim=2
    )[:, :, :-1, :]
    transparency_matrix = torch.cumprod(transparencies, dim=2)

    final_canvas = torch.sum(transparency_matrix * pixel_colors, dim=2)
    final_canvas += canvas * torch.prod(opacities, dim=2)

    return final_canvas


def mse_loss(rendered_image, target_image):
    """
    Calculates the Mean Squared Error between the rendered and target images.
    """
    return ((rendered_image - target_image) ** 2).mean()
