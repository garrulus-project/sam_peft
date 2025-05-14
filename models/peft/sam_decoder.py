import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class SAMDecoder(nn.Module):
    """
    SAM Decoder model only. Keep SAM Encoder frozen.
    """

    def __init__(self, sam_model: Sam, use_dense_embeddings=True):
        super(SAMDecoder, self).__init__()
        self.use_dense_embeddings = use_dense_embeddings

        self.adapter_layer = list(range(len(sam_model.image_encoder.blocks)))

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        if not self.use_dense_embeddings:
            print("Dense embedding is not used, grad update is disabled")
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False

        self.sam = sam_model

    def save_peft_parameters(self, filename: str) -> None:
        """
        Save prompt and mask decoder params
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(
            self.sam, torch.nn.parallel.DistributedDataParallel
        ):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()

        for key, value in state_dict.items():
            if "prompt_encoder" in key:
                prompt_encoder_tensors[key] = value
            if "mask_decoder" in key:
                mask_decoder_tensors[key] = value

        if self.use_dense_embeddings:
            merged_dict = {**prompt_encoder_tensors, **mask_decoder_tensors}
        else:
            merged_dict = {**mask_decoder_tensors}

        torch.save(merged_dict, filename)

    def load_peft_parameters(self, filename: str, device=None) -> None:
        """
        Load mask decoder and prompt encoder param
        """

        assert filename.endswith(".pt") or filename.endswith(".pth")

        if device is not None:
            state_dict = torch.load(filename, map_location=device)
        else:
            state_dict = torch.load(filename)

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
