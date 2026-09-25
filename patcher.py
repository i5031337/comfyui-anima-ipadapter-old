"""DiT block monkey-patching for Anima IP-Adapter cross-attention injection."""

import inspect
import torch.nn.functional as F


def patch_dit_blocks(dit, num_blocks):
    """Hooks cross_attn to capture query and monkey-patches block forward."""
    patched_count = 0
    for i, blk in enumerate(dit.blocks):
        if i >= num_blocks:
            break
        if not getattr(blk, "use_ip_adapter", False):
            continue

        if not getattr(blk, "_ip_hook_installed", False):
            def _make_capture(ref):
                def _hook(_mod, args):
                    ref._x_cross_flat = args[0].detach()
                return _hook
            blk.cross_attn.register_forward_pre_hook(_make_capture(blk))
            blk._ip_hook_installed = True

        if getattr(blk, "_ip_fwd_patched", False):
            continue

        orig_fwd = blk.forward
        if "ip_hidden_states" in inspect.signature(orig_fwd).parameters:
            continue

        def _make_fwd(blk_ref, orig):
            def _patched_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                                 rope_emb_L_1_1_D=None,
                                 adaln_lora_B_T_3D=None,
                                 extra_per_block_pos_emb=None,
                                 transformer_options=None,
                                 **__kwargs):
                result = orig(
                    x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                    rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                    extra_per_block_pos_emb=extra_per_block_pos_emb,
                    transformer_options=transformer_options,
                    **__kwargs,
                )
                if not getattr(blk_ref, "use_ip_adapter", False):
                    return result
                if transformer_options is None:
                    return result
                ip_tok = transformer_options.get("anima_ip_tokens", None)
                if ip_tok is None:
                    return result
                if not hasattr(blk_ref, "_x_cross_flat"):
                    return result

                x_q = blk_ref._x_cross_flat
                ip_tok = ip_tok.to(device=x_q.device, dtype=x_q.dtype)
                B = x_B_T_H_W_D.shape[0]
                T, H, W = x_B_T_H_W_D.shape[1], x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3]
                n_h = blk_ref.cross_attn.n_heads
                h_d = blk_ref.cross_attn.head_dim

                # Ensure adaln_ip is on same device/dtype as emb_B_T_D
                try:
                    gate_ip = blk_ref.adaln_ip(emb_B_T_D)
                except (RuntimeError, TypeError):
                    blk_ref.adaln_ip = blk_ref.adaln_ip.to(device=emb_B_T_D.device, dtype=emb_B_T_D.dtype)
                    gate_ip = blk_ref.adaln_ip(emb_B_T_D)

                scale_mask = (ip_tok.abs().sum(dim=[1, 2]) > 1e-6).to(
                    dtype=result.dtype).reshape(B, 1, 1)

                ip_q = blk_ref.cross_attn.q_proj(x_q).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                ip_q = blk_ref.cross_attn.q_norm(ip_q)
                ip_k = blk_ref.ip_k_proj(ip_tok).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                ip_v = blk_ref.ip_v_proj(ip_tok).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
                ip_kn = getattr(blk_ref, "ip_k_norm", None)
                if ip_kn is not None:
                    ip_k = ip_kn(ip_k)

                ip_attn = F.scaled_dot_product_attention(ip_q, ip_k, ip_v)
                ip_out = ip_attn.permute(0, 2, 1, 3).reshape(B, T * H * W, n_h * h_d)

                gate = (gate_ip * scale_mask).to(dtype=ip_out.dtype)
                ip_delta = (gate.reshape(B, T, 1, 1, -1) * ip_out.reshape(B, T, H, W, -1)).to(dtype=result.dtype)
                result = result + ip_delta
                return result
            return _patched_forward

        blk.forward = _make_fwd(blk, orig_fwd)
        blk._ip_fwd_patched = True
        patched_count += 1

    if patched_count:
        print(f"[AnimaIPAdapter] Patched {patched_count} block forwards for IP cross-attention")
