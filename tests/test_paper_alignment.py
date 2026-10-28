"""Static graph checks; these do not import PyTorch or execute the network."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def tree(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))


def method(class_name, name="forward"):
    cls = next(n for n in tree("net.py").body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned(scope, name):
    return next(n.value for n in ast.walk(scope) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def returned(scope):
    return next(n.value for n in ast.walk(scope) if isinstance(n, ast.Return))


class ManuscriptGraphTests(unittest.TestCase):
    def same_expression(self, actual, expected):
        self.assertEqual(ast.dump(actual, include_attributes=False),
                         ast.dump(ast.parse(expected, mode="eval").body,
                                  include_attributes=False))

    def test_equation_6_updated_base(self):
        self.same_expression(
            assigned(method("BaseDetailSynergisticIntegration"), "detail_refined"),
            "detail * self.base_to_detail(base_refined)")

    def test_equations_10_and_16_direct_boundary_map(self):
        scope = method("BoundaryAwareRoutingBase")
        self.same_expression(assigned(scope, "routed_feat"),
                             "v * final_gate + i * (1.0 - final_gate)")
        self.same_expression(returned(scope), "routed_feat")
        self.assertNotIn("fuse_conv", ast.unparse(method("BoundaryAwareRoutingBase", "__init__")))

    def test_equations_21_to_24_direct_prototype_map(self):
        scope = method("DeepUnfoldedSemanticClustering")
        self.same_expression(
            assigned(scope, "fused_flat"),
            "v_orig_flat * (1.0 - routing_gate) + i_orig_flat * routing_gate")
        self.same_expression(returned(scope), "fused_clustered")
        init = ast.unparse(method("DeepUnfoldedSemanticClustering", "__init__"))
        self.assertNotIn("fuse_conv", init)
        self.assertNotIn("self.act", init)

    def test_equation_33_no_unreported_logit_clipping(self):
        self.same_expression(
            assigned(method("SpatialAdaptiveDynamicFilter"), "w_logits"),
            "self.weight_gen(x).unsqueeze(1)")

    def test_equations_27_to_29_no_cross_residual(self):
        self.same_expression(returned(method("CrossSequenceMambaFusion")), "(v_out, i_out)")
        init = ast.unparse(method("CrossSequenceMambaFusion", "__init__"))
        self.assertNotIn("v_proj", init)
        self.assertNotIn("i_proj", init)

    def test_equations_25_and_26_update_before_csm(self):
        call = next(n for n in ast.walk(method("AdvancedBaseFusion"))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "tcmf_fusion")
        self.same_expression(call.args[0], "v_route + cluster_feat")
        self.same_expression(call.args[1], "i_route + cluster_feat")

    def test_equation_30_matching_modality_projections(self):
        self.same_expression(
            returned(method("AdvancedBaseFusion")),
            "self.fuse_conv(torch.cat([self.v_proj(v_mamba), self.i_proj(i_mamba)], dim=1))")
        init = ast.unparse(method("AdvancedBaseFusion", "__init__"))
        for extra in ("mica_amp", "self.cca", "refine_block", "nn.Sequential"):
            self.assertNotIn(extra, init)
        self.assertIn("nn.Conv2d(dim * 2, dim, 3, padding=1)", init)

    def test_equation_36_multiscale_residual_only(self):
        scope = method("MultiScaleResidualBlock")
        assignments = [n.value for n in ast.walk(scope) if isinstance(n, ast.Assign)]
        expected = ast.dump(ast.parse("self.fuse(out) + x", mode="eval").body,
                            include_attributes=False)
        self.assertTrue(any(ast.dump(n, include_attributes=False) == expected
                            for n in assignments))
        self.same_expression(returned(scope), "out")
        self.assertNotIn("star_refiner", ast.unparse(scope))
        self.assertIn("MultiScaleResidualBlock(dim)",
                      ast.unparse(method("SpatialChannelDetailFusion", "__init__")))

    def test_equation_43_all_correlation_terms(self):
        self.same_expression(
            assigned(tree("train.py"), "loss_decomp"),
            "cc_loss_D ** 2 + 0.5 * (1.0 - cc_loss_B) + 0.5 * (cc_loss_V_BD ** 2 + cc_loss_I_BD ** 2)")

    def test_training_constants_and_global_schedule(self):
        scope = tree("train.py")
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in scope.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and isinstance(n.value, ast.Constant)}
        expected = {
            "num_epochs": 100, "epoch_gap": 30, "lr": 1e-4, "batch_size": 2,
            "weight_decay": 0, "optim_step": 4, "optim_gamma": 0.5,
            "clip_grad_norm_value": 0.01, "coeff_decomp": 2, "coeff_tv": 5,
            "coeff_ssim": 5, "coeff_ortho": 5,
            "architecture_version": "paper-equations-v1",
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(values[key], value)
        self.same_expression(assigned(scope, "epoch_lr"),
                             "lr * optim_gamma ** (epoch // optim_step)")

    def test_fixed_kernels_excluded_from_stage_toggle(self):
        scope = tree("train.py")
        self.assertIn("if parameter.requires_grad",
                      ast.unparse(assigned(scope, "trainable_parameters")))
        calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "requires_grad_"]
        self.assertEqual(len(calls), 1)
        self.same_expression(calls[0], "parameter.requires_grad_(enabled)")

    def test_syntax_without_executing_models(self):
        for path in sorted(ROOT.rglob("*.py")):
            if ".git" in path.parts or "tmp" in path.relative_to(ROOT).parts:
                continue
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
