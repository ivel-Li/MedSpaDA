"""L2 clipping with float64 norm aggregation and explicit failure diagnostics."""
import math


def finite_clip(named_parameters, max_norm, torch):
    pairs = [(name, p) for name, p in named_parameters if p.grad is not None]
    if not pairs:
        return 0.0, []
    norms = torch.stack([torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64)
                         for _, p in pairs])
    total = torch.linalg.vector_norm(norms)
    norm_value = float(total)
    if not math.isfinite(norm_value):
        bad = [name for (name, _), norm in zip(pairs, norms.tolist()) if not math.isfinite(norm)]
        raise RuntimeError(f"non-finite gradient elements: {bad[:12]}")
    diagnostics = []
    if norm_value > 1e6:
        values, indices = norms.topk(min(5, len(pairs)))
        for value, index in zip(values.tolist(), indices.tolist()):
            name, parameter = pairs[index]
            diagnostics.append({"parameter": name, "gradient_norm": value,
                                "gradient_max_abs": float(parameter.grad.detach().abs().max()),
                                "parameter_max_abs": float(parameter.detach().abs().max())})
    coefficient = (max_norm / (total + 1e-6)).clamp(max=1.0)
    for _, parameter in pairs:
        parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return norm_value, diagnostics
