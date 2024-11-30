import torch
from torch import nn
from functools import partial
import copy

# core quantization method (simulated quantization)
@torch.no_grad()
def pseudo_quantize_tensor(w, n_bits, q_group_size):
    assert n_bits > 0
    assert q_group_size > 0
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2, f"Invalid weight shape: {w.shape}"
    # Set nans to zero
    w[torch.isnan(w)] = 0.0
    w[torch.isinf(w)] = 0.0

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    try:
        max_val = w.amax(dim=1, keepdim=True)
        assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
        min_val = w.amin(dim=1, keepdim=True)
        assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1
    except RuntimeError as e:
        print("\nError in amax or amin")
        print(w.shape)
        # count nans and infs
        print(f"Nans: {torch.isnan(w).sum()}")
        print(f"Infs: {torch.isinf(w).sum()}")
        raise e
    

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bits - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


@torch.no_grad()
def pseudo_quantize_tensor_with_protection(w, n_bits, q_group_size, q_protection_ratio, q_protection_scale):
    assert q_protection_ratio >= 0.0 - 1e-5
    assert q_protection_scale >= 0.0 - 1e-5
    if q_protection_ratio <= 1e-5:
        # No activation protection
        return pseudo_quantize_tensor(w, n_bits, q_group_size)
    importance = sum(w.abs()).float()
    k = int(q_protection_ratio * len(importance))
    _, outlier_mask = torch.topk(importance, k=k, largest=True)
    if q_protection_scale >= 1.0:
        w.data[:, outlier_mask] *= q_protection_scale
        w = pseudo_quantize_tensor(w, n_bits, q_group_size)
        w.data[:, outlier_mask] /= q_protection_scale
    else:
        outlier = w.data[:, outlier_mask]
        w = pseudo_quantize_tensor(w, n_bits, q_group_size)
        w.data[:, outlier_mask] = outlier
    return w


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits):
    # w: (out_features, in_features)
    assert n_bits > 0
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits):
    # w: (out_features, in_features)
    assert n_bits > 0
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits):
    assert n_bits > 0
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits):
    assert n_bits > 0
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class WQAQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        n_bits=-1,
        q_group_size=-1,
        q_protect=False,
        q_protection_ratio=-1.0,
        q_protection_scale=-1.0
    ):
        assert n_bits > 0
        assert q_group_size >= 0
        super().__init__()
        self.n_bits = n_bits
        self.q_group_size = q_group_size
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)
        if q_protect:
            assert q_protection_ratio > -1e-5
            assert q_protection_scale > -1e-5
            assert q_group_size > 0
            self.act_quant_name = f"protected_group_quant_{q_group_size}"
            self.act_quant = partial(
                pseudo_quantize_tensor_with_protection,
                n_bits=n_bits, q_group_size=q_group_size,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
        elif q_group_size > 0:
            self.act_quant_name = f"group_quant_{q_group_size}"
            self.act_quant = partial(pseudo_quantize_tensor, n_bits=self.n_bits, q_group_size=self.q_group_size)
        elif act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WQAQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False,
        n_bits=-1, q_group_size=-1, q_protect=False, q_protection_ratio=-1.0, q_protection_scale=-1.0
    ):
        assert isinstance(module, torch.nn.Linear)
        assert n_bits > 0
        assert q_group_size >= 0
        new_module = WQAQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            n_bits=n_bits,
            q_group_size=q_group_size,
            q_protect=q_protect,
            q_protection_ratio=q_protection_ratio,
            q_protection_scale=q_protection_scale,
        )
        if q_protect:
            assert q_protection_ratio > -1e-5
            assert q_protection_scale > -1e-5
            assert q_group_size > 0
            new_module.weight = copy.deepcopy(module.weight) # Already quantized by AWQ.
            new_module.weight_quant_name = f"protected_group_quant_{q_group_size}"
        elif q_group_size > 0:
            new_module.weight = pseudo_quantize_tensor(
                module.weight, n_bits=n_bits, q_group_size=q_group_size
            )
            new_module.weight_quant_name = f"group_quant_{q_group_size}"
        elif weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=n_bits
            )
            new_module.weight_quant_name = weight_quant
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=n_bits
            )
            new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"WALinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True,
    n_bits=-1, q_group_size=-1, q_protect=False, q_protection_ratio=-1.0, q_protection_scale=-1.0
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )
    assert n_bits > 0
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = WQAQLinear.from_float(
                m.fc1, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.fc2 = WQAQLinear.from_float(
                m.fc2, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.out_proj = WQAQLinear.from_float(
                m.out_proj, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
    return model


def quantize_llama_like(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False,
    n_bits=-1, q_group_size=-1, q_protect=False, q_protection_ratio=-1.0, q_protection_scale=-1.0
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    assert n_bits > 0

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = WQAQLinear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.up_proj = WQAQLinear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.down_proj = WQAQLinear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
            m.o_proj = WQAQLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant,
                n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
                q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False, n_bits=-1
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBlockSparseTop2MLP,
    )
    assert n_bits > 0
    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBlockSparseTop2MLP):
            m.w1 = WQAQLinear.from_float(
                m.w1, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
            m.w2 = WQAQLinear.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
            m.w3 = WQAQLinear.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
            )
            m.o_proj = WQAQLinear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = WQAQLinear.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
    return model


def quantize_falcon(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, n_bits=-1
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )
    assert n_bits > 0

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = WQAQLinear.from_float(
                m.dense_h_to_4h, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
            m.dense_4h_to_h = WQAQLinear.from_float(
                m.dense_4h_to_h, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = WQAQLinear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                n_bits=n_bits,
            )
            m.dense = WQAQLinear.from_float(
                m.dense, weight_quant=weight_quant, act_quant=act_quant, n_bits=n_bits
            )
    return model


def quantize_model(
    model, weight_quant="per_channel", act_quant="per_token",
    quantize_bmm_input=False,
    n_bits=-1, q_group_size=-1, q_protect=False, q_protection_ratio=-1.0, q_protection_scale=-1.0
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    assert n_bits > 0
    assert q_group_size >= 0
    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
            q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
        )
    elif isinstance(model, (LlamaPreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            n_bits=n_bits, q_group_size=q_group_size, q_protect=q_protect,
            q_protection_ratio=q_protection_ratio, q_protection_scale=q_protection_scale,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            n_bits=n_bits,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            n_bits=n_bits,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
