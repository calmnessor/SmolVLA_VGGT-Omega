import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class _FloatDenseHead(nn.Module):
    def __init__(self, dense_head):
        super().__init__()
        self.dense_head = dense_head

    def forward(self, aggregated_tokens_list, images, patch_token_start):
        tokens = [None if token is None else token.float() for token in aggregated_tokens_list]
        return self.dense_head(tokens, images=images.float(), patch_token_start=patch_token_start)

def prepare_vggt_images(images: list[Tensor], resolution: int) -> Tensor:
    """Convert two SmolVLA images from [-1,1] to [B,2,3,R,R] in [0,1]."""
    if len(images) != 2:
        raise ValueError(f"VGGT E1 expects exactly two camera images, got {len(images)}")
    if any(image.ndim != 4 or image.shape[1] != 3 for image in images):
        raise ValueError("Each VGGT camera image must have shape [B, 3, H, W]")
    if images[0].shape[0] != images[1].shape[0]:
        raise ValueError("VGGT camera images must have the same batch size")
    resized = [F.interpolate(image, size=(resolution, resolution), mode="bicubic", align_corners=False) for image in images]
    return torch.stack(resized, dim=1).add(1).div(2).clamp(0, 1)

class FrozenVGGTSceneEncoder(nn.Module):
    output_dim = 2048
    def __init__(self, checkpoint_path: str, code_path: str, image_resolution: int = 512, num_register_tokens: int = 16, device: str = "cuda", enable_depth: bool = False):
        super().__init__()
        checkpoint, source = Path(checkpoint_path).expanduser(), Path(code_path).expanduser()
        if not checkpoint.is_file(): raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
        if not (source / "vggt_omega").is_dir(): raise FileNotFoundError(f"VGGT source package not found under: {source}")
        if str(source) not in sys.path: sys.path.insert(0, str(source))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.models.heads import DenseHead
        model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        if enable_depth:
            rng_state = torch.random.get_rng_state()
            model.dense_head = DenseHead(dim_in=2 * 1024, patch_size=16)
            torch.random.set_rng_state(rng_state)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        keys = set(model.state_dict())
        missing, unexpected = model.load_state_dict({k:v for k,v in state.items() if k in keys}, strict=False)
        if missing or unexpected: raise RuntimeError(f"VGGT checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        if enable_depth:
            model.dense_head = _FloatDenseHead(model.dense_head)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(device=device, dtype=dtype).eval()
        if enable_depth:
            model.dense_head.dense_head.float()
        for param in model.parameters(): param.requires_grad = False
        self.__dict__["_model"] = model
        self.image_resolution, self.num_register_tokens = image_resolution, num_register_tokens
        self.depth_enabled = enable_depth
    def train(self, mode=True):
        super().train(mode); self._model.eval(); return self
    @torch.no_grad()
    def forward(self, images: list[Tensor], include_depth: bool = False) -> Tensor | dict[str, Tensor]:
        x = prepare_vggt_images(images, self.image_resolution)
        device = next(self._model.parameters()).device
        if include_depth and not self.depth_enabled:
            raise RuntimeError("VGGT depth output was not enabled at construction")
        if include_depth:
            predictions = self._model(x.to(device))
        else:
            dense_head = self._model.dense_head
            self._model.dense_head = None
            try:
                predictions = self._model(x.to(device))
            finally:
                self._model.dense_head = dense_head
        registers = predictions["camera_and_register_tokens"][:, :, 1:1+self.num_register_tokens]
        b, n, r, d = registers.shape
        if r != self.num_register_tokens or d != self.output_dim: raise RuntimeError(f"Unexpected VGGT register shape: {tuple(registers.shape)}")
        registers = registers.reshape(b, n * r, d)
        if not include_depth:
            return registers
        return {"registers": registers.detach(), "depth": predictions["depth"].detach(), "depth_conf": predictions["depth_conf"].detach()}
