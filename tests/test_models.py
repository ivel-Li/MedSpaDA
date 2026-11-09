import unittest

import torch

from medspada.adapters import MonaSliceAdapter, _FacTQKV, _FacTShared
from medspada.refinement import GeneralistSpatialResidualC0
from medspada.refinement_loss import selected_q0, c0_window_loss
from medspada.resolution import HighResolutionSegmentationHead


class ModelTests(unittest.TestCase):
    def test_adapter_starts_as_identity(self):
        x = torch.randn(3, 4, 4, 8)
        for kind in ("mona_parallel", "mona_serial"):
            adapter = MonaSliceAdapter(8, 4, kind=kind, dropout=0)
            torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)

    def test_depth_branch_uses_adjacent_slice_and_can_be_disabled(self):
        torch.manual_seed(42)
        adapter = MonaSliceAdapter(8, 4, kind="mona_parallel", dropout=0).eval()
        torch.nn.init.normal_(adapter.up.weight)
        a = torch.randn(3, 4, 4, 8); b = a.clone(); b[1] += 1
        self.assertGreater(float((adapter(a)[0]-adapter(b)[0]).abs().max().detach()), 0)
        adapter.set_depth_residual_enabled(False)
        torch.testing.assert_close(adapter(a)[0], adapter(b)[0])

    def test_fact_changes_qv_but_preserves_key_projection(self):
        shared = _FacTShared(8, 2)
        torch.nn.init.normal_(shared.up.weight)
        base = torch.nn.Linear(8, 24)
        wrapped = _FacTQKV(base, shared, 2, scale=1, dropout=0)
        x = torch.randn(2, 3, 8)
        expected = base(x); actual = wrapped(x)
        torch.testing.assert_close(actual[..., 8:16], expected[..., 8:16], rtol=0, atol=0)
        self.assertGreater(float((actual[..., :8]-expected[..., :8]).abs().max().detach()), 0)
        self.assertGreater(float((actual[..., 16:]-expected[..., 16:]).abs().max().detach()), 0)

    def test_vsr_starts_as_identity_and_keeps_base_gradient(self):
        vsr = GeneralistSpatialResidualC0()
        x = torch.randn(2, 12, 12, requires_grad=True)
        refined, residual = vsr(x, torch.tensor([0.8, 0.9]))
        torch.testing.assert_close(refined, x, rtol=0, atol=0)
        torch.testing.assert_close(residual, torch.zeros_like(x), rtol=0, atol=0)
        refined.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        self.assertGreater(float(vsr.residual.weight.grad.abs().sum()), 0)

    def test_selection_uses_native_object_times_presence_score(self):
        output = {"pred_logits": torch.tensor([[[0.], [2.], [1.]]]),
                  "presence_logit_dec": torch.tensor([0.]),
                  "pred_masks": torch.arange(12).reshape(1, 3, 2, 2).float()}
        mask, score, index = selected_q0(output, 3, torch)
        self.assertEqual(index.item(), 1)
        torch.testing.assert_close(score, torch.sigmoid(torch.tensor([2.]))*0.5)
        torch.testing.assert_close(mask, output["pred_masks"][:, 1])

    def test_volume_loss_handles_all_negative_targets(self):
        logits = torch.zeros(1, 3, 4, 4, requires_grad=True)
        loss, _ = c0_window_loss(logits, logits*0, torch.ones(1, 3),
                                torch.zeros_like(logits), 0.6, torch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hr1_keeps_interpolation_differentiable(self):
        base = torch.nn.Module(); base.no_dec = False; base.aux_masks = False; base.d_model = 4
        head = HighResolutionSegmentationHead(base, half_size=6, full_size=8)
        low = torch.randn(1, 2, 3, 3, requires_grad=True)
        head._base_outputs = lambda *args: (None, None, low, low[:, :1], None)
        output = head(None, None, None, encoder_hidden_states=torch.zeros(1))
        expected = torch.nn.functional.interpolate(torch.nn.functional.interpolate(low,
            size=(6, 6), mode="bilinear", align_corners=False), size=(8, 8), mode="bilinear", align_corners=False)
        torch.testing.assert_close(output["pred_masks"], expected)
        output["pred_masks"].sum().backward()
        self.assertGreater(float(low.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
