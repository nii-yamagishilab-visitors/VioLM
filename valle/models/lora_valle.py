import logging
from contextlib import contextmanager
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class InProjLinear(nn.Linear):
    """
    Subclass of nn.Linear so MultiheadAttention builds in_proj_linear (an nn.Module)
    instead of fused in_proj_weight. Used only when building the model inside
    use_attn_qkv_linear(); no changes to valle.py or activation.py.
    """
    pass


class SeparateQKVLinearAttention(nn.Module):
    """
    Drop-in replacement for valle.modules.activation.MultiheadAttention that
    exposes q_proj, k_proj, v_proj, out_proj as separate nn.Linear instances,
    enabling individual LoRA adaptation of each projection.

    Accepts the same constructor signature as MultiheadAttention so it can be
    installed transparently via use_attn_separate_qkv_linear().

    linear1_cls is accepted but ignored — q/k/v are always plain nn.Linear so
    that LoRA wrappers can replace them individually. linear2_cls is forwarded
    to out_proj.

    forward() is implemented manually (does NOT call F.multi_head_attention_forward
    for the input projections) so that LoRA wrappers on q_proj/k_proj/v_proj are
    exercised through their own forward() methods.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        linear1_cls=nn.Linear,   # accepted but ignored for q/k/v
        linear2_cls=nn.Linear,   # forwarded to out_proj
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Separate projection linears — individually replaceable by LoRA
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        # out_proj uses linear2_cls so the existing attn_out LoRA path also works
        self.out_proj = linear2_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
            self.bias_v = nn.Parameter(
                torch.empty((1, 1, embed_dim), **factory_kwargs)
            )
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            constant_(self.q_proj.bias, 0.0)
            constant_(self.k_proj.bias, 0.0)
            constant_(self.v_proj.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
            xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.bias_k is not None or self.add_zero_attn:
            raise NotImplementedError(
                "SeparateQKVLinearAttention does not support add_bias_kv or "
                "add_zero_attn (not used in VALLE/VALLF)."
            )

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(0, 1)
                else:
                    query, key = query.transpose(0, 1), key.transpose(0, 1)
                    value = key
            else:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)

        # query: [L, N, E]  key/value: [S, N, E_k/E_v]
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]

        # --- project through LoRA-able linears ---
        q = self.q_proj(query)   # [L, N, E]
        k = self.k_proj(key)     # [S, N, E]
        v = self.v_proj(value)   # [S, N, E]

        # Reshape: [len, N, E] -> [N*H, len, head_dim]
        scaling = float(self.head_dim) ** -0.5
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product
        attn_weights = torch.bmm(q * scaling, k.transpose(1, 2))  # [N*H, L, S]

        # attn_mask: [L, S] or [N*H, L, S]
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attn_mask

        # key_padding_mask: [N, S] -> broadcast to [N*H, 1, S]
        if key_padding_mask is not None:
            kpm = key_padding_mask.view(bsz, 1, 1, src_len)
            kpm = kpm.expand(-1, self.num_heads, -1, -1)
            kpm = kpm.reshape(bsz * self.num_heads, 1, src_len)
            if key_padding_mask.dtype == torch.bool:
                attn_weights.masked_fill_(kpm, float("-inf"))
            else:
                attn_weights = attn_weights + kpm

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.bmm(attn_weights, v)  # [N*H, L, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, self.embed_dim)
        )

        # Output projection (also LoRA-able)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            w = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                w = w.mean(dim=1)
        else:
            w = None

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, w


@contextmanager
def use_attn_qkv_linear():
    """
    Context manager: build the model inside this to get Q/K/V as nn.Linear
    (in_proj_linear) so apply_lora_to_valle(..., target_modules="...attn_qkv")
    can replace them. Patches transformer layer defaults at runtime; no edits
    to valle.py, activation.py, or models/__init__.py.
    """
    from valle.modules import transformer

    orig_enc_init = transformer.TransformerEncoderLayer.__init__
    orig_dec_init = transformer.TransformerDecoderLayer.__init__

    def patched_enc_init(self, *args, **kwargs):
        kwargs.setdefault("linear1_self_attention_cls", InProjLinear)
        return orig_enc_init(self, *args, **kwargs)

    def patched_dec_init(self, *args, **kwargs):
        kwargs.setdefault("linear1_self_attention_cls", InProjLinear)
        return orig_dec_init(self, *args, **kwargs)

    transformer.TransformerEncoderLayer.__init__ = patched_enc_init
    transformer.TransformerDecoderLayer.__init__ = patched_dec_init
    try:
        yield
    finally:
        transformer.TransformerEncoderLayer.__init__ = orig_enc_init
        transformer.TransformerDecoderLayer.__init__ = orig_dec_init


@contextmanager
def use_attn_separate_qkv_linear():
    """
    Context manager: build the model inside this to get separate q_proj, k_proj,
    v_proj nn.Linear instances in each attention module, enabling individual LoRA
    adaptation via apply_lora_to_valle(..., target_modules="attn_q+attn_k+attn_v").

    Patches valle.modules.transformer.MultiheadAttention in-place for the duration
    of the context so TransformerEncoderLayer / TransformerDecoderLayer instantiate
    SeparateQKVLinearAttention instead. Zero changes to activation.py or transformer.py.

    Use with 'attn', 'attn_q', 'attn_k', 'attn_v', or 'all' target_modules.
    Mutually exclusive with use_attn_qkv_linear().
    """
    from valle.modules import transformer as _transformer_module

    orig_mha = _transformer_module.MultiheadAttention
    _transformer_module.MultiheadAttention = SeparateQKVLinearAttention
    try:
        yield
    finally:
        _transformer_module.MultiheadAttention = orig_mha


def _import_loralib():
    """
    Import loralib lazily with a clear error message if it's missing.
    """
    try:
        import loralib as lora  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "LoRA support for VALLE requires the `loralib` package.\n"
            "Install it with:\n"
            "  pip install loralib\n"
        ) from exc
    return lora


def _parse_targets(target_modules: str) -> Set[str]:
    """
    Parse a target modules string like 'ffn+attn_out' or 'ffn,attn'.

    Expansion rules:
      'attn'     -> {'attn_out', 'attn_q', 'attn_k', 'attn_v'}
                    Requires model built with use_attn_separate_qkv_linear().
      'all'      -> {'ffn', 'attn_out', 'attn_q', 'attn_k', 'attn_v'}
                    Requires model built with use_attn_separate_qkv_linear().
      'attn_qkv' -> fused Q/K/V (legacy). Requires use_attn_qkv_linear().
                    Mutually exclusive with attn_q/k/v.

    Combine with '+' or ',', e.g. 'ffn+attn' or 'ffn+attn_out+attn_q+attn_v'.
    """
    parts: Iterable[str] = []
    for sep in ["+", ","]:
        if sep in target_modules:
            parts = target_modules.split(sep)
            break
    if not parts:
        parts = (target_modules,)
    raw = {p.strip().lower() for p in parts if p.strip()}
    out: Set[str] = set()
    for t in raw:
        if t == "attn":
            out.update({"attn_out", "attn_q", "attn_k", "attn_v"})
        elif t == "all":
            out.update({"ffn", "attn_out", "attn_q", "attn_k", "attn_v"})
        else:
            out.add(t)
    return out


def copy_fused_in_proj_into_model(model: nn.Module, state_dict: dict) -> None:
    """
    Copy fused in_proj_weight / in_proj_bias from state_dict into model's
    in_proj_linear (when present). Use after load_state_dict(..., strict=False)
    when the checkpoint was saved with fused in_proj and the model was built
    with use_attn_qkv_linear() (in_proj_linear).
    """
    for name, module in model.named_modules():
        if not hasattr(module, "in_proj_linear"):
            continue
        prefix = name + "." if name else ""
        key_w = prefix + "in_proj_weight"
        key_b = prefix + "in_proj_bias"
        if key_w in state_dict:
            module.in_proj_linear.weight.copy_(state_dict[key_w])
        if key_b in state_dict and module.in_proj_linear.bias is not None:
            module.in_proj_linear.bias.copy_(state_dict[key_b])


def copy_fused_in_proj_into_separate_qkv(
    model: nn.Module, state_dict: dict
) -> None:
    """
    Split and copy fused in_proj_weight / in_proj_bias from a pretrained checkpoint
    into the separate q_proj / k_proj / v_proj of a model built with
    use_attn_separate_qkv_linear(). Call after load_state_dict(..., strict=False).

    PyTorch convention for in_proj_weight row ordering:
        rows [0   : E  ] -> Q
        rows [E   : 2*E] -> K
        rows [2*E : 3*E] -> V
    Same split applies to in_proj_bias.
    """
    for name, module in model.named_modules():
        if not isinstance(module, SeparateQKVLinearAttention):
            continue
        prefix = name + "." if name else ""
        key_w = prefix + "in_proj_weight"
        key_b = prefix + "in_proj_bias"
        if key_w not in state_dict:
            continue
        fused_w = state_dict[key_w]  # [3*E, E]
        E = module.embed_dim
        assert fused_w.shape == (3 * E, E), (
            f"Expected in_proj_weight shape [{3*E}, {E}], got {fused_w.shape}"
        )
        with torch.no_grad():
            module.q_proj.weight.copy_(fused_w[0:E])
            module.k_proj.weight.copy_(fused_w[E : 2 * E])
            module.v_proj.weight.copy_(fused_w[2 * E : 3 * E])
        if key_b in state_dict and module.q_proj.bias is not None:
            fused_b = state_dict[key_b]  # [3*E]
            with torch.no_grad():
                module.q_proj.bias.copy_(fused_b[0:E])
                module.k_proj.bias.copy_(fused_b[E : 2 * E])
                module.v_proj.bias.copy_(fused_b[2 * E : 3 * E])


def _make_lora_linear(
    base_linear: nn.Linear,
    r: int,
    alpha: int,
    dropout: float,
    lora,
) -> nn.Module:
    """
    Create a LoRA-wrapped Linear initialized to behave like `base_linear`.
    """
    lora_linear = lora.Linear(
        base_linear.in_features,
        base_linear.out_features,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )

    # Copy the existing weights so that, with zeroed LoRA adapters, the
    # effective weight is identical to the original model.
    with torch.no_grad():
        lora_linear.weight.copy_(base_linear.weight)
        if base_linear.bias is not None and lora_linear.bias is not None:
            lora_linear.bias.copy_(base_linear.bias)

    return lora_linear


def apply_lora_to_valle(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: str = "ffn+attn_out",
    freeze_base: bool = True,
) -> None:
    """
    In-place add LoRA adapters to a VALLE model.

    This function works without modifying valle.py, activation.py, or transformer.py.
    It operates by inspecting and replacing selected nn.Linear layers inside the
    transformer blocks.

    Args:
        model:
            The instantiated VALLE model (or a module that contains it).
        r:
            LoRA rank.
        alpha:
            LoRA scaling factor.
        dropout:
            LoRA dropout probability.
        target_modules:
            Which sub-layers to adapt. Combination of:
            - 'ffn'      : feed-forward linear1 and linear2.
            - 'attn_out' : attention output projection (O).
            - 'attn_q'   : Q projection (requires use_attn_separate_qkv_linear()).
            - 'attn_k'   : K projection (requires use_attn_separate_qkv_linear()).
            - 'attn_v'   : V projection (requires use_attn_separate_qkv_linear()).
            - 'attn'     : shorthand for attn_out + attn_q + attn_k + attn_v.
            - 'attn_qkv' : fused Q/K/V (legacy; requires use_attn_qkv_linear()).
            - 'all'      : ffn + attn_out + attn_q + attn_k + attn_v.
            Combine with '+' or ',', e.g. 'ffn+attn' or 'ffn+attn_out+attn_q'.
        freeze_base:
            If True, freezes all existing parameters before adding LoRA, so that
            only LoRA parameters are trainable.
    """
    lora = _import_loralib()
    targets = _parse_targets(target_modules)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    num_ffn = 0
    num_attn_out = 0
    num_attn_q = 0
    num_attn_k = 0
    num_attn_v = 0
    num_attn_qkv = 0

    for module in model.modules():
        # --- FFN: TransformerEncoderLayer / TransformerDecoderLayer ---
        if "ffn" in targets:
            if hasattr(module, "linear1") and isinstance(
                getattr(module, "linear1"), nn.Linear
            ) and not isinstance(getattr(module, "linear1"), lora.Linear):
                base = module.linear1
                module.linear1 = _make_lora_linear(
                    base, r=r, alpha=alpha, dropout=dropout, lora=lora
                )
                num_ffn += 1

            if hasattr(module, "linear2") and isinstance(
                getattr(module, "linear2"), nn.Linear
            ) and not isinstance(getattr(module, "linear2"), lora.Linear):
                base = module.linear2
                module.linear2 = _make_lora_linear(
                    base, r=r, alpha=alpha, dropout=dropout, lora=lora
                )
                num_ffn += 1

        # --- Attention output projection (O) ---
        if "attn_out" in targets:
            for attn_attr in ("self_attn", "multihead_attn"):
                if not hasattr(module, attn_attr):
                    continue
                attn = getattr(module, attn_attr)
                if hasattr(attn, "out_proj") and isinstance(
                    attn.out_proj, nn.Linear
                ) and not isinstance(attn.out_proj, lora.Linear):
                    attn.out_proj = _make_lora_linear(
                        attn.out_proj, r=r, alpha=alpha, dropout=dropout, lora=lora
                    )
                    num_attn_out += 1

        # --- Separate Q projection ---
        if "attn_q" in targets:
            for attn_attr in ("self_attn", "multihead_attn"):
                if not hasattr(module, attn_attr):
                    continue
                attn = getattr(module, attn_attr)
                if hasattr(attn, "q_proj") and isinstance(
                    attn.q_proj, nn.Linear
                ) and not isinstance(attn.q_proj, lora.Linear):
                    attn.q_proj = _make_lora_linear(
                        attn.q_proj, r=r, alpha=alpha, dropout=dropout, lora=lora
                    )
                    num_attn_q += 1

        # --- Separate K projection ---
        if "attn_k" in targets:
            for attn_attr in ("self_attn", "multihead_attn"):
                if not hasattr(module, attn_attr):
                    continue
                attn = getattr(module, attn_attr)
                if hasattr(attn, "k_proj") and isinstance(
                    attn.k_proj, nn.Linear
                ) and not isinstance(attn.k_proj, lora.Linear):
                    attn.k_proj = _make_lora_linear(
                        attn.k_proj, r=r, alpha=alpha, dropout=dropout, lora=lora
                    )
                    num_attn_k += 1

        # --- Separate V projection ---
        if "attn_v" in targets:
            for attn_attr in ("self_attn", "multihead_attn"):
                if not hasattr(module, attn_attr):
                    continue
                attn = getattr(module, attn_attr)
                if hasattr(attn, "v_proj") and isinstance(
                    attn.v_proj, nn.Linear
                ) and not isinstance(attn.v_proj, lora.Linear):
                    attn.v_proj = _make_lora_linear(
                        attn.v_proj, r=r, alpha=alpha, dropout=dropout, lora=lora
                    )
                    num_attn_v += 1

        # --- Legacy fused Q/K/V projection ---
        # activation.py's MultiheadAttention passes in_proj_weight (a raw tensor) directly
        # to F.multi_head_attention_forward, bypassing in_proj_linear.forward(). We must
        # register a forward pre-hook to recompute in_proj_weight = base + B@A*scaling
        # before each forward so the LoRA delta is actually applied.
        if "attn_qkv" in targets:
            for attn_attr in ("self_attn", "multihead_attn"):
                if not hasattr(module, attn_attr):
                    continue
                attn = getattr(module, attn_attr)
                if hasattr(attn, "in_proj_linear") and isinstance(
                    attn.in_proj_linear, nn.Linear
                ) and not isinstance(attn.in_proj_linear, lora.Linear):
                    new_lin = _make_lora_linear(
                        attn.in_proj_linear, r=r, alpha=alpha, dropout=dropout, lora=lora
                    )
                    attn.in_proj_linear = new_lin
                    # activation.py registers in_proj_weight as Parameter (because
                    # `self.in_proj_weight = self.in_proj_linear.weight` triggers
                    # Module.__setattr__ which adds Parameters to _parameters).
                    # The forward hook needs to overwrite it with a plain tensor each
                    # forward pass (to inject the LoRA delta), which Module.__setattr__
                    # rejects for registered Parameters. Delete from _parameters first
                    # so the hook's plain-tensor assignment uses the normal __dict__ path.
                    if 'in_proj_weight' in attn._parameters:
                        del attn._parameters['in_proj_weight']
                    attn.in_proj_weight = new_lin.weight.data
                    if new_lin.bias is not None:
                        if 'in_proj_bias' in attn._parameters:
                            del attn._parameters['in_proj_bias']
                        attn.in_proj_bias = new_lin.bias.data
                    # Register pre-hook: before each attention forward, inject the
                    # effective weight (base + B@A*scaling) into in_proj_weight so
                    # F.multi_head_attention_forward picks it up and gradients flow through.
                    def _make_qkv_hook(lin_ref):
                        def _hook(mod, _args):
                            lin = mod.in_proj_linear
                            if isinstance(lin, lora.Linear) and not lin.merged and lin.r > 0:
                                mod.in_proj_weight = (
                                    lin.weight + (lin.lora_B @ lin.lora_A) * lin.scaling
                                )
                            else:
                                mod.in_proj_weight = lin.weight
                        return _hook
                    # Remove any existing hook to avoid stacking on double-apply
                    if hasattr(attn, "_lora_qkv_hook"):
                        attn._lora_qkv_hook.remove()
                    attn._lora_qkv_hook = attn.register_forward_pre_hook(
                        _make_qkv_hook(new_lin)
                    )
                    num_attn_qkv += 1

    logging.info(
        f"Applied LoRA to VALLE model "
        f"(rank={r}, alpha={alpha}, dropout={dropout}, "
        f"ffn={num_ffn}, attn_out={num_attn_out}, "
        f"attn_q={num_attn_q}, attn_k={num_attn_k}, attn_v={num_attn_v}, "
        f"attn_qkv={num_attn_qkv})"
    )
