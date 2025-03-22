import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import opencood.hypes_yaml.yaml_utils as yaml_utils


def init_t_xy(end_x, end_y):
    """
    Initialize x and y coordinates for a 2D grid.

    Args:
        end_x (int): The length of the x-axis (e.g., image height).
        end_y (int): The length of the y-axis (e.g., image width).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - t_x (torch.Tensor): Tensor containing x coordinates.
            - t_y (torch.Tensor): Tensor containing y coordinates.
    """
    # Create a 1D tensor with values from 0 to (end_x * end_y - 1)
    t = torch.arange(end_x * end_y, dtype=torch.float32)

    # Calculate x coordinates by taking the remainder of t divided by end_x
    # This effectively maps each position to its x-coordinate in the grid
    t_x = (t.remainder(end_x)).float()

    # Calculate y coordinates by performing integer division of t by end_x
    # This maps each position to its y-coordinate in the grid
    t_y = t.div(end_x, rounding_mode="floor").float()

    # Return the x and y coordinate tensors
    return t_x, t_y


def compute_axial_cis(dim, end_x, end_y, theta=100.0, device="cuda"):
    """
    Compute axial complex rotary embeddings for 2D spatial dimensions.

    Args:
        dim (int): The embedding dimension. Must be a multiple of 4.
        end_x (int): The length of the x-axis (e.g., image height).
        end_y (int): The length of the y-axis (e.g., image width).
        theta (float, optional): A scaling factor controlling the frequency distribution. Defaults to 100.0.
        device (str, optional): The device to store the tensor ('cuda' or 'cpu'). Defaults to "cuda".

    Returns:
        torch.Tensor:
            - A tensor of shape (end_x * end_y, dim // 2) containing complex rotary embeddings.
    """
    # Generate a range of frequencies with exponential decay controlled by theta
    # torch.arange(0, dim, 4) creates a tensor [0, 4, 8, ..., dim-4]
    # [: dim // 4] ensures the tensor length is dim // 4
    # The frequencies are scaled by theta raised to the power of (index / dim)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 4, dtype=torch.float32)[: dim // 4] / dim)
    )

    # Initialize x and y coordinates using the init_t_xy function
    t_x, t_y = init_t_xy(end_x, end_y)

    # Compute the outer product of t_x and freqs to get frequency-scaled x coordinates
    freqs_x = torch.outer(t_x, freqs)

    # Compute the outer product of t_y and freqs to get frequency-scaled y coordinates
    freqs_y = torch.outer(t_y, freqs)

    # Generate complex rotary embeddings for the x-axis using polar coordinates
    # torch.polar(magnitude, angle) creates a complex tensor from magnitude and angle
    # Here, magnitude is set to 1 (unit circle), and angle is freqs_x
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)

    # Generate complex rotary embeddings for the y-axis using polar coordinates
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    # Concatenate the x and y rotary embeddings along the last dimension
    # This combines the embeddings from both spatial axes
    axial_cis = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    # Move the concatenated rotary embeddings to the specified device (e.g., GPU)
    axial_cis = axial_cis.to(device)

    # Return the final rotary embeddings tensor
    return axial_cis


def reshape_for_broadcast(freqs_cis, x):
    """
    Reshape the freqs_cis tensor to be compatible for broadcasting with tensor x.

    Args:
        freqs_cis (torch.Tensor): The rotary embedding tensor to be reshaped.
        x (torch.Tensor): The tensor with which freqs_cis will be broadcasted.

    Returns:
        torch.Tensor: The reshaped freqs_cis tensor ready for broadcasting.

    Raises:
        ValueError: If the shape of freqs_cis does not match any expected pattern with x.
    """
    # Get the number of dimensions of tensor x
    ndim = x.ndim

    # Ensure that tensor x has at least 2 dimensions
    assert ndim > 1, "Input tensor x must have at least 2 dimensions."

    # Check if freqs_cis has the same shape as the last two dimensions of x
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        # Create a shape list with 1s for all dimensions except the last two
        shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]

    # Check if freqs_cis has the same shape as the last three dimensions of x
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        # Create a shape list with 1s for all dimensions except the last three
        shape = [1] * (ndim - 3) + [x.shape[-3], x.shape[-2], x.shape[-1]]

    # Check if freqs_cis has the same shape as the first and last dimensions of x
    elif freqs_cis.shape == (x.shape[1], x.shape[-1]):
        # Initialize a shape list with 1s for all dimensions
        shape = [1] * ndim
        # Set the second dimension to match the second dimension of x
        shape[1] = x.shape[1]
        # Set the last dimension to match the last dimension of x
        shape[-1] = x.shape[-1]

    # If freqs_cis does not match any expected shape, raise an error
    else:
        raise ValueError("Shape of freqs_cis does not match x in any expected pattern.")

    # Reshape freqs_cis to the computed shape
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to query and key tensors using complex multiplication.

    Args:
        xq (torch.Tensor): Query tensor of shape (..., C).
        xk (torch.Tensor): Key tensor of shape (..., C).
        freqs_cis (torch.Tensor): Rotary embedding tensor to be applied.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors with the same shape as input.
    """
    # Convert query and key tensors to complex numbers by pairing the last dimension
    # Assume the last dimension is even and can be reshaped to (..., C//2, 2) representing real and imaginary parts
    xq_complex, xk_complex = (
        torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)),
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)),
    )

    # Reshape freqs_cis to be broadcastable with the complex query tensor
    freqs_cis_broadcast = reshape_for_broadcast(freqs_cis, xq_complex)

    # Apply rotary embeddings by complex multiplication
    # This rotates the query and key vectors in the complex plane
    xq_rotated = torch.view_as_real(xq_complex * freqs_cis_broadcast).flatten(-2)
    xk_rotated = torch.view_as_real(xk_complex * freqs_cis_broadcast).flatten(-2)

    # Convert the rotated tensors back to the original data type of xq and xk
    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)


class CausalSpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % 32 == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.pose_tokens_num = config.token_size_dict["pose_tokens_size"]
        self.img_tokens_num = config.token_size_dict["img_tokens_size"]
        self.yaw_token_size = config.token_size_dict["yaw_token_size"]
        self.total_tokens_num = config.token_size_dict["total_tokens_size"]
        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num
        self.freqs_cis_singlescale = compute_axial_cis(
            dim=config.n_embd // self.n_head,
            end_x=self.patch_size[0],
            end_y=self.patch_size[1],
            theta=1000.0,
            device=config.device,
        )

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if T > self.pose_tokens_num + self.yaw_token_size:
            q_B_scale2_d = q[:, :, self.pose_tokens_num + self.yaw_token_size :, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num + self.yaw_token_size :, :]
            q_out, k_out = apply_rotary_emb(
                q_B_scale2_d,
                k_B_scale2_d,
                freqs_cis=self.freqs_cis_singlescale[
                    : T - self.pose_tokens_num - self.yaw_token_size
                ],
            )
            q = torch.cat(
                [q[:, :, 0 : self.pose_tokens_num + self.yaw_token_size, :], q_out],
                dim=2,
            )
            k = torch.cat(
                [k[:, :, 0 : self.pose_tokens_num + self.yaw_token_size, :], k_out],
                dim=2,
            )
        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        y = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.attn_dropout_rate,
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )
        y = self.resid_drop(self.proj(y))
        return y


class CausalSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class SpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.pose_tokens_num = config.token_size_dict["pose_tokens_size"]
        self.img_tokens_num = config.token_size_dict["img_tokens_size"]
        self.yaw_token_size = config.token_size_dict["yaw_token_size"]
        self.total_tokens_num = config.token_size_dict["total_tokens_size"]
        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num

        self.freqs_cis_singlescale = compute_axial_cis(
            dim=config.n_embd // self.n_head,
            end_x=self.patch_size[0],
            end_y=self.patch_size[1],
            theta=1000.0,
            device=config.device,
        )

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if T > self.pose_tokens_num + self.yaw_token_size:
            q_B_scale2_d = q[:, :, self.pose_tokens_num + self.yaw_token_size :, :]
            k_B_scale2_d = k[:, :, self.pose_tokens_num + self.yaw_token_size :, :]
            q_out, k_out = apply_rotary_emb(
                q_B_scale2_d,
                k_B_scale2_d,
                freqs_cis=self.freqs_cis_singlescale[
                    : T - self.pose_tokens_num - self.yaw_token_size
                ],
            )
            q = torch.cat(
                [q[:, :, 0 : self.pose_tokens_num + self.yaw_token_size, :], q_out],
                dim=2,
            )
            k = torch.cat(
                [k[:, :, 0 : self.pose_tokens_num + self.yaw_token_size, :], k_out],
                dim=2,
            )
        if attn_mask.ndim == 3:
            attn_mask = attn_mask[:, None, :, :]
        y = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.attn_dropout_rate,
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )
        y = self.resid_drop(self.proj(y))
        return y


class SpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTimeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.to(q.dtype),
                dropout_p=self.attn_dropout_rate,
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )
        y = self.resid_drop(self.proj(y))
        return y


class CausalTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalTimeSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_time_block = CausalTimeBlock(config)
        self.space_block = SpaceBlock(config)

    def forward(self, x, time_attn_mask, space_attn_mask):
        b, f, l, c = x.shape
        x = rearrange(x, "b f l c -> (b l) f c")
        x = self.causal_time_block(x, time_attn_mask)
        x = rearrange(x, "(b l) f c -> b f l c", b=b, l=l, f=f)
        space_attn_mask = space_attn_mask.unsqueeze(1)
        space_attn_mask = space_attn_mask.repeat(1, x.shape[1], 1, 1)
        x = rearrange(x, "b f l c -> (b f) l c", b=b, f=f)
        space_attn_mask = rearrange(space_attn_mask, "b f l c -> (b f) l c")
        x = self.space_block(x, space_attn_mask)
        x = rearrange(x, "(b f) l c -> b f l c", b=b, f=f)
        return x


class FeatureTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        block_size,
        n_layer=[3, 3],
        n_head=8,
        n_embd=1024,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        n_unmasked=0,
        condition_frames=3,
        latent_size=(24, 88),
        L=24 * 88,
        token_size_dict=None,
        device="cuda",
    ):
        super(FeatureTimeSeriesTransformer, self).__init__()
        config = GPTConfig(
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
            patch_size=latent_size,
            condition_frames=condition_frames,
            token_size_dict=token_size_dict,
            device=device,
        )

        self.n_embd = config.n_embd  # Assume n_embd is C
        self.num_layers = n_layer
        self.total_token_size = config.token_size_dict["total_tokens_size"]

        # # Feature embedding, mapping C to d_model
        # self.feature_proj = nn.Linear(config.n_embd, self.d_model)

        # Stacking spatiotemporal Transformer layers
        self.causal_time_space_blocks = nn.ModuleList(
            [CausalTimeSpaceBlock(config) for _ in range(self.num_layers[0])]
        )
        self.causal_space_blocks = nn.Sequential(
            *[CausalSpaceBlock(config) for _ in range(self.num_layers[1])]
        )

        condition_frames = config.condition_frames
        matrix = torch.tril(torch.ones(condition_frames, condition_frames))
        time_causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        time_causal_mask = torch.where(matrix == 1, 0, time_causal_mask)
        # [[0, -inf, -inf], 
        #  [0,   0,  -inf], 
        #  [0,   0,    0]]  shape(condition_frames, condition_frames)
        self.mask_time = time_causal_mask.to(config.device)

        matrix_1 = torch.tril(torch.ones(self.total_token_size, self.total_token_size))
        seq_causal_mask = torch.where(matrix_1 == 0, float("-inf"), matrix_1)
        seq_causal_mask = torch.where(matrix_1 == 1, 0, seq_causal_mask)
        # [[0, -inf, -inf], 
        #  [0,   0,  -inf], 
        #  [0,   0,    0]]  shape(total_token_size, total_token_size)
        self.mask_ar = seq_causal_mask.contiguous().to(config.device)

        mask_spatial = torch.ones(self.total_token_size, self.total_token_size)
        # [[1, 1, 1],
        #  [1, 1, 1],
        #  [1, 1, 1]]  shape(total_token_size, total_token_size)
        self.mask_spatial = mask_spatial.to(config.device)

    def forward(self, x):
        """
        x: (B, T, L, C)
        """
        B, T, L, C = x.size()
        assert C == self.n_embd, (
            f"Expected input feature dimension {self.n_embd}, but got {C}"
        )

        space_attn_mask = self.mask_spatial.unsqueeze(0).repeat(B, 1, 1)
        mask_ar = self.mask_ar.unsqueeze(0).repeat(B * T, 1, 1)
        for block in self.causal_time_space_blocks:
            x = block(x, self.mask_time, space_attn_mask)  # (B, T, H*W, d_model)
        x = rearrange(x, "B T L C -> (B T) L C", B=B, T=T)
        for block in self.causal_space_blocks:
            x = block(x, mask_ar)
        x = rearrange(x, "(B T) L C-> B T L C", B=B, T=T)
        # Take features of the last time step
        x = torch.squeeze(x[:, -1, :, :], dim=1)  # (B, H*W, d_model)

        return x

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

if __name__ == "__main__":
    # Input data
    B, T, L, C = (
        1,
        10,
        2112,
        1024,
    )  # Example batch size, timesteps, spatial size, channels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = "/home/UAP_attack/uap/configs/uap.yaml"
    config = yaml_utils.load_yaml(config_file, None)
    args = config["model"]["world_model"]
    condition_frames = args["condition_frames"]
    total_token_size = args["total_token_size"]
    feature_token_size = args["feature_token_size"]
    pose_token_size = args["pose_token_size"]
    yaw_token_size = args["yaw_token_size"]
    feature_size = args["feature_size"]
    downsample_factor = args["downsample_factor"]
    latent_size = [int(x / downsample_factor) for x in feature_size]
    input_dim = args["input_dim"]

    token_size_dict = {
        "img_tokens_size": feature_token_size,
        "pose_tokens_size": pose_token_size,
        "yaw_token_size": yaw_token_size,
        "total_tokens_size": total_token_size,
    }
    transformer_config = {
        "block_size": condition_frames * (total_token_size),
        "n_layer": args["n_layer"],
        "n_head": args["n_head"],
        "n_embd": args["embedding_dim"],
        "condition_frames": condition_frames,
        "token_size_dict": token_size_dict,
        "latent_size": latent_size,
        "L": feature_token_size,
        "device": device,
    }
    model = FeatureTimeSeriesTransformer(**transformer_config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    input_tensor = torch.randn(B, T, L, C)  
    input_tensor = input_tensor.to(device)
    start = time.time()
    # Forward pass
    # with torch.no_grad():
    #     output = model(input_tensor)
    with torch.no_grad():
        output = model(input_tensor)  # (B, H, W, C)

    # Define loss function
    # criterion = nn.MSELoss()

    # # Compute loss
    # loss = criterion(output, input_tensor[:, -1, :, :, :])

    # # Backward pass
    # loss.backward()

    # Print peak memory usage
    print(
        f"Peak memory usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB"
    )
    end = time.time()
    print(f"Time elapsed: {end - start:.6f} s")
    print(output.shape)
