"""Tests for safe YAML config loading — guards against pseudo-comment keys.

Pseudo-comment keys (e.g. ``'# 目标': value`` in YAML) are a common user
error when editing config files.  They are meant to be YAML comments
(``# 目标``) but are accidentally written as quoted keys, which YAML
parses as real dictionary entries.  :func:`_safe_load_yaml` detects and
strips them with a warning.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def safe_load():
    """Import helper with logging captured."""
    from subway_defect.train.configs import _safe_load_yaml
    return _safe_load_yaml


class TestSafeLoadYaml:
    """Unit tests for :func:`_safe_load_yaml`."""

    def test_normal_yaml(self, safe_load) -> None:
        """Normal YAML without pseudo-comments — loaded as-is."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("epochs: 120\nimgsz: 1024\nbatch: 16\n")
            f.flush()
            path = Path(f.name)

        try:
            cfg = safe_load(path)
            assert cfg == {"epochs": 120, "imgsz": 1024, "batch": 16}
        finally:
            path.unlink(missing_ok=True)

    def test_strips_pseudo_comment_keys(self, safe_load, caplog) -> None:
        """Keys starting with '#' are stripped with a warning."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "'# ===== Phase 3 =====': ''\n"
                "'# 目标': 学习工业纹理\n"
                "'# 使用方式': 'data: ...'\n"
                "epochs: 120\n"
                "imgsz: 1024\n"
                "nc: 1\n"
                "'# 验收': mAP50 达标\n"
            )
            f.flush()
            path = Path(f.name)

        try:
            with caplog.at_level(logging.WARNING):
                cfg = safe_load(path)

            # Pseudo-comment keys stripped
            assert "epochs" in cfg
            assert "imgsz" in cfg
            assert "nc" in cfg
            assert "# 目标" not in cfg
            assert "# ===== Phase 3 =====" not in cfg
            assert "# 验收" not in cfg
            # Values preserved for real keys
            assert cfg["epochs"] == 120
            # Warning emitted
            assert any("Pseudo-comment keys" in r.message for r in caplog.records)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_returns_empty(self, safe_load) -> None:
        """Missing file → empty dict (no crash)."""
        cfg = safe_load(Path("/nonexistent/config_xyz.yaml"))
        assert cfg == {}

    def test_non_dict_yaml_returns_empty(self, safe_load) -> None:
        """YAML that parses to a list → empty dict."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("- item1\n- item2\n")
            f.flush()
            path = Path(f.name)

        try:
            cfg = safe_load(path)
            assert cfg == {}
        finally:
            path.unlink(missing_ok=True)

    def test_empty_yaml_returns_empty(self, safe_load) -> None:
        """Empty YAML file → empty dict (not None)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            cfg = safe_load(path)
            assert cfg == {}
        finally:
            path.unlink(missing_ok=True)

    def test_stage_configs_load_clean(self, safe_load) -> None:
        """All real pretrain stage configs must load without pseudo-comment warnings."""
        import logging as _logging
        from subway_defect.train.configs import _CONFIG_DIR

        pretrain_dir = _CONFIG_DIR / "train" / "pretrain"
        for yaml_file in sorted(pretrain_dir.glob("*.yaml")):
            # Verify each loads cleanly — no pseudo-comment keys
            cfg = safe_load(yaml_file)
            assert isinstance(cfg, dict), f"{yaml_file.name}: not a dict"
            # No key should start with '#'
            pseudo = [k for k in cfg if str(k).startswith("#")]
            assert not pseudo, (
                f"{yaml_file.name}: found pseudo-comment keys {pseudo}. "
                f"Replace \"'# key': value\" with \"# key\" (true YAML comment)."
            )


class TestConfigIntegration:
    """Integration tests for config loading functions."""

    def test_load_train_config_returns_yolo_args(self) -> None:
        """load_train_config returns valid dict with expected keys."""
        from subway_defect.train.configs import load_train_config

        cfg = load_train_config("warmup")
        assert "epochs" in cfg
        assert "imgsz" in cfg
        assert "batch" in cfg
        # Must not contain pseudo-comment keys
        assert not any(str(k).startswith("#") for k in cfg)

    def test_load_pretrain_config(self) -> None:
        """load_pretrain_config for existing stages returns clean configs."""
        from subway_defect.train.configs import load_pretrain_config

        for stage in [
            "stage1_neck_head_warmup",
            "stage2_scale_adaptation",
            "stage3_short_finetune",
            "stage4_hard_negative",
        ]:
            cfg = load_pretrain_config(stage)
            assert isinstance(cfg, dict)
            assert "epochs" in cfg
            assert not any(str(k).startswith("#") for k in cfg)

    def test_check_dict_alignment_rejects_pseudo_comments(self) -> None:
        """check_dict_alignment gives a helpful message for pseudo-comment keys."""
        from subway_yolo.cfg import get_cfg, check_dict_alignment
        from subway_defect import PROJECT_ROOT

        # Use a model YAML to get a valid base config
        model_yaml = PROJECT_ROOT / "subway_defect" / "models" / "yolo11s-EMA-SimAM.yaml"
        base = vars(get_cfg(str(model_yaml)))  # SimpleNamespace → dict
        bad_cfg = {
            "epochs": 1,
            "# 目标": "test",  # pseudo-comment — should trigger message
            "train": "images/train",  # not a YOLO arg
        }
        with pytest.raises(SyntaxError) as exc_info:
            check_dict_alignment(base, bad_cfg)

        msg = str(exc_info.value)
        assert "Pseudo-comment" in msg or "# 目标" in msg
