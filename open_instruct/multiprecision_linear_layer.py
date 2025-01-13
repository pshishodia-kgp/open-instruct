import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum, auto

class LinearLayerType(Enum):
    floatlm = auto()
    trilm = auto()
    values17 = auto()

def replace_linear_with_multiprecision(model: nn.Module, num_trilm_matrix_scales: int):
    for name, layer in model.named_modules():
        # all linear layers except lm_head and embed_tokens. embed_tokens is of type Embedding.
        if  (not isinstance(layer, torch.nn.Linear)) or ('lm_head' in name):
            continue
        multiprecision_layer = MultiPrecisionLinearLayer(
            in_dims=layer.in_features,
            out_dims=layer.out_features,
            bias=layer.bias,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
            linear_type=LinearLayerType.trilm,
            num_trilm_matrix_scales=num_trilm_matrix_scales
        )
        multiprecision_layer.linear.weight = layer.weight
        
        if layer.bias:
            multiprecision_layer.linear.bias = layer.bias

class MultiPrecisionLinearLayer(nn.Module):
    # TODO: Fix ⁠ num_trilm_matrix_scales ⁠ to mimick model parallelism (i.e. take row parallel and column parallel into account)
    def __init__(self, in_dims: int, out_dims: int, bias: bool, device: torch.device, dtype: torch.dtype, linear_type: LinearLayerType, num_trilm_matrix_scales: int):
        super().__init__()
        self.typecast_to = torch.float32 # If you modify this, also modify optionally_ternarize_single_matrix() function in convert_olmo_to_hf_new.py
        self.linear_type = linear_type
        self.num_trilm_matrix_scales = num_trilm_matrix_scales
        
        assert self.linear_type in [LinearLayerType.floatlm, LinearLayerType.trilm, LinearLayerType.values17], self.linear_type
        if self.linear_type in [LinearLayerType.trilm, LinearLayerType.values17]:
            assert self.num_trilm_matrix_scales > 0, f"{self.num_trilm_matrix_scales=} should be > 0 for {layer_type=}"
            assert in_dims % self.num_trilm_matrix_scales == 0, f"{self.num_trilm_matrix_scales=} should divide {in_dims=} for {layer_type=}"
            assert out_dims % self.num_trilm_matrix_scales == 0, f"{self.num_trilm_matrix_scales=} should divide {out_dims=} for {layer_type=}"
        elif self.linear_type == LinearLayerType.floatlm:
            assert self.num_trilm_matrix_scales == 0, f"{self.num_trilm_matrix_scales=} should be zero for {layer_type=}"
        else:
            raise NotImplementedError
        self.linear = nn.Linear(in_dims, out_dims, bias=bias, device=device, dtype=dtype)
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        target_weight = self.linear.weight
        target_bias = self.linear.bias
        ######################
        ####### WARNING ######
        ######################
        # If you modify this, also modify optionally_ternarize_single_matrix() function in convert_olmo_to_hf_new.py
        if self.linear_type == LinearLayerType.trilm:
            # Reshape according to num scales and typecast to FP32
            target_weight_groupped_reshaped_fp32 = target_weight.reshape(self.num_trilm_matrix_scales, -1).to(self.typecast_to)
            # Get Scale and Quantize
            target_weight_scale_fp32 = 1 / target_weight_groupped_reshaped_fp32.abs().mean(1).clamp_(min=1e-5).unsqueeze(-1)
            target_weight_quantized_fp32 = (target_weight_groupped_reshaped_fp32 * target_weight_scale_fp32).round().clamp_(-1, 1) / target_weight_scale_fp32
            # Reshape back and back to original dtype.
            target_weight_quantized = target_weight_quantized_fp32.reshape(target_weight.shape).to(target_weight.dtype)
            # Apply STE
            target_weight = target_weight + (target_weight_quantized - target_weight).detach()
        elif self.linear_type == LinearLayerType.floatlm:
            pass
        else:
            raise NotImplementedError
        return F.linear(x, target_weight, bias=self.linear.bias)