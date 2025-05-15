import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class AdaptMLP(nn.Module):
    r"""
    Adapter in parallel with MLP
    """

    def __init__(
        self,
        scaling: int,
        mlp: nn.Module,
        down_fn: nn.Module,
        up_fn: nn.Module,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.mlp = mlp
        self.down_fn = down_fn
        self.up_fn = up_fn
        self.act = act_layer()
        self.scaling = scaling

    def forward(self, x):
        mlp_out = self.mlp(x)
        adapter = self.up_fn(self.act(self.down_fn(x)))
        out = self.scaling * adapter + mlp_out

        return out


class AdapterAttention(nn.Module):
    r"""
    Adapter after attention
    """

    def __init__(
        self,
        attn: nn.Module,
        down_fn: nn.Module,
        up_fn: nn.Module,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.attn = attn
        self.down_fn = down_fn
        self.up_fn = up_fn
        self.act = act_layer()
        self.shortcut = True

    def forward(self, x):
        attn_out = self.attn(x)
        out = attn_out + self.up_fn(self.act(self.down_fn(attn_out)))

        return out


class AdapterSAM(nn.Module):
    """
    Based on Wu et al., Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation, 2023
    """

    def __init__(
        self,
        sam_model: Sam,
        middle_dim: int,
        scaling_factor: int,
        use_dense_embeddings=True,
    ):
        super(AdapterSAM, self).__init__()
        self.use_dense_embeddings = use_dense_embeddings

        assert middle_dim > 0

        self.adapter_layer = list(
            range(len(sam_model.image_encoder.blocks))
        )  # Only apply adapter to the image encoder by default

        # initialize the adapter layers
        self.w_down_attn = []
        self.w_up_attn = []
        self.w_down_mlp = []
        self.w_up_mlp = []

        # disable training image encoder at first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # disable training dense embedding and no mask dense embedding
        if not self.use_dense_embeddings:
            print("Dense embedding is not used, grad update is disabled")
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False

        # inject adapters to the image encoder
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.adapter_layer:
                continue
            atten = blk.attn
            mlp = blk.mlp
            self.dim = blk.attn.qkv.in_features
            w_down_linear_attn = nn.Linear(self.dim, middle_dim, bias=True)
            w_up_linear_attn = nn.Linear(middle_dim, self.dim, bias=True)
            w_down_linear_mlp = nn.Linear(self.dim, middle_dim, bias=True)
            w_up_linear_mlp = nn.Linear(middle_dim, self.dim, bias=True)
            self.w_down_attn.append(w_down_linear_attn)
            self.w_up_attn.append(w_up_linear_attn)
            self.w_down_mlp.append(w_down_linear_mlp)
            self.w_up_mlp.append(w_up_linear_mlp)
            blk.attn = AdapterAttention(atten, w_down_linear_attn, w_up_linear_attn)
            blk.mlp = AdaptMLP(
                scaling_factor,
                mlp,
                w_down_linear_mlp,
                w_up_linear_mlp,
            )

        self.sam = sam_model

    def save_peft_parameters(self, filename: str) -> None:
        """
        Save peft parameters to a file.
        Args:
            filename (str): The path to the file to save the parameters to.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_down_attn)
        a_tensors = {
            f"w_a_{i:03d}": self.w_down_attn[i].weight for i in range(num_layer)
        }
        a_bias = {
            f"w_a_{i:03d}_bia": self.w_down_attn[i].bias for i in range(num_layer)
        }
        b_tensors = {f"w_b_{i:03d}": self.w_up_attn[i].weight for i in range(num_layer)}
        b_bias = {f"w_b_{i:03d}_bia": self.w_up_attn[i].bias for i in range(num_layer)}
        c_tensors = {
            f"w_c_{i:03d}": self.w_down_mlp[i].weight for i in range(num_layer)
        }
        c_bias = {f"w_c_{i:03d}_bia": self.w_down_mlp[i].bias for i in range(num_layer)}
        d_tensors = {f"w_d_{i:03d}": self.w_up_mlp[i].weight for i in range(num_layer)}
        d_bias = {f"w_d_{i:03d}_bia": self.w_up_mlp[i].bias for i in range(num_layer)}

        # prompt_encoder_tensors = {}
        # mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        # for key, value in state_dict.items():
        #     if 'prompt_encoder' in key:
        #         prompt_encoder_tensors[key] = value
        #     if 'mask_decoder' in key:
        #         mask_decoder_tensors[key] = value

        # prompt embedding and mask decoder tensors
        pe_md_tensors = {}
        for key, value in state_dict.items():
            if "prompt_encoder" in key and self.use_dense_embeddings:
                pe_md_tensors[key] = value
            if "mask_decoder" in key:
                pe_md_tensors[key] = value

        merged_dict = {
            **a_tensors,
            **b_tensors,
            **c_tensors,
            **d_tensors,
            **a_bias,
            **b_bias,
            **c_bias,
            **d_bias,
            **pe_md_tensors,
        }
        torch.save(merged_dict, filename)

    def load_peft_parameters(self, filename: str, device=None) -> None:
        """
        Load peft parameters from a file.
        Args:
            filename (str): The path to the file containing the parameters.
            device (torch.device, optional): The device to load the parameters on. If None, loads on CPU.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        if device is not None:
            state_dict = torch.load(filename, map_location=device)
        else:
            state_dict = torch.load(filename)

        for i, w_down_linear_attn in enumerate(self.w_down_attn):
            saved_key = f"w_a_{i:03d}"
            saved_key_bia = f"w_a_{i:03d}_bia"
            saved_tensor = state_dict[saved_key]
            saved_tensor_bia = state_dict[saved_key_bia]
            w_down_linear_attn.weight = Parameter(saved_tensor)
            w_down_linear_attn.bias = Parameter(saved_tensor_bia)

        for i, w_up_linear_attn in enumerate(self.w_up_attn):
            saved_key = f"w_b_{i:03d}"
            saved_key_bia = f"w_b_{i:03d}_bia"
            saved_tensor = state_dict[saved_key]
            saved_tensor_bia = state_dict[saved_key_bia]
            w_up_linear_attn.weight = Parameter(saved_tensor)
            w_up_linear_attn.bias = Parameter(saved_tensor_bia)

        for i, w_down_linear_mlp in enumerate(self.w_down_mlp):
            saved_key = f"w_c_{i:03d}"
            saved_key_bia = f"w_c_{i:03d}_bia"
            saved_tensor = state_dict[saved_key]
            saved_tensor_bia = state_dict[saved_key_bia]
            w_down_linear_mlp.weight = Parameter(saved_tensor)
            w_down_linear_mlp.bias = Parameter(saved_tensor_bia)

        for i, w_up_linear_mlp in enumerate(self.w_up_mlp):
            saved_key = f"w_d_{i:03d}"
            saved_key_bia = f"w_d_{i:03d}_bia"
            saved_tensor = state_dict[saved_key]
            saved_tensor_bia = state_dict[saved_key_bia]
            w_up_linear_mlp.weight = Parameter(saved_tensor)
            w_up_linear_mlp.bias = Parameter(saved_tensor_bia)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        if self.use_dense_embeddings:
            prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
            prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
            prompt_encoder_new_state_dict = {
                k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)
            }
            sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {
            k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)
        }
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(
            batched_input,
            multimask_output,
            image_size,
            use_dense_embeddings=self.use_dense_embeddings,
        )
