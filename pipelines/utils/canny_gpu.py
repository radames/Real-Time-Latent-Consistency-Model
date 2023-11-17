import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

class SobelOperator(nn.Module):
    def __init__(self, device="cuda"):
        super(SobelOperator, self).__init__()
        self.device = device
        self.edge_conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )
        self.edge_conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )

        sobel_kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=self.device
        )
        sobel_kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=self.device
        )

        self.edge_conv_x.weight = nn.Parameter(sobel_kernel_x.view((1, 1, 3, 3)))
        self.edge_conv_y.weight = nn.Parameter(sobel_kernel_y.view((1, 1, 3, 3)))

    @torch.no_grad()
    def forward(self, image: Image.Image, low_threshold: float, high_threshold: float):
        # Convert PIL image to PyTorch tensor
        image_gray = image.convert("L")
        image_tensor = ToTensor()(image_gray).unsqueeze(0).to(self.device)

        # Compute gradients
        edge_x = self.edge_conv_x(image_tensor)
        edge_y = self.edge_conv_y(image_tensor)
        edge = torch.sqrt(edge_x**2 + edge_y**2)

        # Apply thresholding
        edge = edge / edge.max()  # Normalize to 0-1
        edge[edge >= high_threshold] = 1.0
        edge[edge <= low_threshold] = 0.0

        # Convert the result back to a PIL image
        return ToPILImage()(edge.squeeze(0).cpu())
