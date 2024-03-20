import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image


class SobelOperator(nn.Module):
    SOBEL_KERNEL_X = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    SOBEL_KERNEL_Y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    )

    def __init__(self, device="cuda"):
        super(SobelOperator, self).__init__()
        self.device = device
        self.edge_conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )
        self.edge_conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )
        self.edge_conv_x.weight = nn.Parameter(
            self.SOBEL_KERNEL_X.view((1, 1, 3, 3)).to(self.device)
        )
        self.edge_conv_y.weight = nn.Parameter(
            self.SOBEL_KERNEL_Y.view((1, 1, 3, 3)).to(self.device)
        )

    @torch.no_grad()
    def forward(
        self,
        image: Image.Image,
        low_threshold: float,
        high_threshold: float,
        output_type="pil",
    ) -> Image.Image | torch.Tensor | tuple[Image.Image, torch.Tensor]:
        # Convert PIL image to PyTorch tensor
        image_gray = image.convert("L")
        image_tensor = ToTensor()(image_gray).unsqueeze(0).to(self.device)

        # Compute gradients
        edge_x = self.edge_conv_x(image_tensor)
        edge_y = self.edge_conv_y(image_tensor)
        edge = torch.sqrt(torch.square(edge_x) + torch.square(edge_y))

        # Apply thresholding
        edge.div_(edge.max())  # Normalize to 0-1 (in-place operation)
        edge[edge >= high_threshold] = 1.0
        edge[edge <= low_threshold] = 0.0

        # Convert the result back to a PIL image
        if output_type == "pil":
            return ToPILImage()(edge.squeeze(0).cpu())
        elif output_type == "tensor":
            return edge
        elif output_type == "pil,tensor":
            return ToPILImage()(edge.squeeze(0).cpu()), edge


class ScharrOperator(nn.Module):
    SCHARR_KERNEL_X = torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]
    )
    SCHARR_KERNEL_Y = torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]
    )

    def __init__(self, device="cuda"):
        super(ScharrOperator, self).__init__()
        self.device = device
        self.edge_conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )
        self.edge_conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            self.device
        )
        self.edge_conv_x.weight = nn.Parameter(
            self.SCHARR_KERNEL_X.view((1, 1, 3, 3)).to(self.device)
        )
        self.edge_conv_y.weight = nn.Parameter(
            self.SCHARR_KERNEL_Y.view((1, 1, 3, 3)).to(self.device)
        )

    @torch.no_grad()
    def forward(
        self,
        image: Image.Image,
        low_threshold: float,
        high_threshold: float,
        output_type="pil",
        invert: bool = False,
    ) -> Image.Image | torch.Tensor | tuple[Image.Image, torch.Tensor]:
        # Convert PIL image to PyTorch tensor
        image_gray = image.convert("L")
        image_tensor = ToTensor()(image_gray).unsqueeze(0).to(self.device)

        # Compute gradients
        edge_x = self.edge_conv_x(image_tensor)
        edge_y = self.edge_conv_y(image_tensor)
        edge = torch.abs(edge_x) + torch.abs(edge_y)

        # Apply thresholding
        edge.div_(edge.max())  # Normalize to 0-1 (in-place operation)
        edge[edge >= high_threshold] = 1.0
        edge[edge <= low_threshold] = 0.0
        if invert:
            edge = 1 - edge

        # Convert the result back to a PIL image
        if output_type == "pil":
            return ToPILImage()(edge.squeeze(0).cpu())
        elif output_type == "tensor":
            return edge
        elif output_type == "pil,tensor":
            return ToPILImage()(edge.squeeze(0).cpu()), edge
