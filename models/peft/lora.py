import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class LoRAQKV(nn.Module):
    """
    Apply LoRA to QV layers.
    ToDo: add options to apply to all QKV layers.
    Args:
        qkv (nn.Module): The original QKV layer.
        linear_a_q (nn.Module): The linear layer for query.
        linear_b_q (nn.Module): The linear layer for query.
        linear_a_v (nn.Module): The linear layer for value.
        linear_b_v (nn.Module): The linear layer for value.
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRASAM(nn.Module):
    """
    Apply LoRA to the image encoder of SAM.
    Args:
        sam_model (Sam): The SAM model to be adapted.
        r (int): The rank of the LoRA matrices.
        lora_layer (list, optional): The layers to apply LoRA to. If None, all layers are used.
        use_dense_embeddings (bool): Whether to use dense embeddings or not.
    """

    def __init__(
        self, sam_model: Sam, r: int, lora_layer=None, use_dense_embeddings=True
    ):
        super(LoRASAM, self).__init__()
        self.use_dense_embeddings = use_dense_embeddings

        assert r > 0

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        # initialize LoRA matrices AB
        self.w_As = []
        self.w_Bs = []

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
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = LoRAQKV(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.sam = sam_model

    def save_peft_parameters(self, filename: str) -> None:
        r"""
        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        # prompt embedding and mask decoder tensors
        pe_md_tensors = {}
        for key, value in state_dict.items():
            if "prompt_encoder" in key and self.use_dense_embeddings:
                pe_md_tensors[key] = value
            if "mask_decoder" in key:
                pe_md_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **pe_md_tensors}
        torch.save(merged_dict, filename)

    def load_peft_parameters(self, filename: str, device=None) -> None:
        r"""
        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        if device is not None:
            state_dict = torch.load(filename, map_location=device)
        else:
            state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

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

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(
            batched_input,
            multimask_output,
            image_size,
            use_dense_embeddings=self.use_dense_embeddings,
        )
