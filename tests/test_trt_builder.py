import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from whisper_s2t.backends.tensorrt import engine_builder as builder


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(builder.importlib.metadata, "version", return_value="1.2.1"),
            patch.object(
                builder.torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
            patch.object(
                builder.torch.cuda, "get_device_name", return_value="RTX 4070 Ti"
            ),
        ]
        for mock in self.patches:
            mock.start()
            self.addCleanup(mock.stop)

    def test_decoder_position_limit_and_invalid_options(self):
        config = builder.TRTBuilderConfig(max_input_len=4, max_output_len=448)
        self.assertEqual(config.max_input_len + config.max_output_len, 448)
        for kwargs in (
            {"max_input_len": 448},
            {"max_batch_size": 0},
            {"dtype": "int8"},
            {"quantization": "fp8"},
            {"use_weight_only_enc": True},
        ):
            with self.assertRaises(ValueError):
                builder.TRTBuilderConfig(**kwargs)

    def test_cache_identity_includes_gpu_and_build_limits(self):
        original = builder.TRTBuilderConfig().identifier()
        self.assertNotEqual(
            original, builder.TRTBuilderConfig(max_batch_size=1).identifier()
        )
        self.assertNotEqual(
            original, builder.TRTBuilderConfig(quantization="int8").identifier()
        )
        with patch.object(builder.torch.cuda, "get_device_name", return_value="L4"):
            self.assertNotEqual(original, builder.TRTBuilderConfig().identifier())

    def test_loader_rejects_legacy_or_wrong_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trt_build_args.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Legacy"):
                builder.load_trt_build_config(tmp)
            config = vars(builder.TRTBuilderConfig())
            config["gpu_name"] = "L4"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "GPU model"):
                builder.load_trt_build_config(tmp)

    def test_int8_converter_receives_weight_only_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "small.pt"
            checkpoint.touch()
            with patch.object(builder, "SAVE_DIR", tmp), patch.object(
                builder, "download_model", return_value=(str(checkpoint), "unused")
            ), patch.object(
                builder.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "convert"),
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    builder.build_trt_engine(
                        "small", builder.TRTBuilderConfig(quantization="int8")
                    )
                command = run.call_args.args[0]
                self.assertIn("--use_weight_only", command)
                self.assertEqual(
                    command[command.index("--weight_only_precision") + 1], "int8"
                )

    def test_failed_conversion_does_not_publish_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = builder.TRTBuilderConfig()
            checkpoint = Path(tmp) / "small.pt"
            checkpoint.touch()
            tokenizer = Path(tmp) / "tokenizer.json"
            tokenizer.write_text("{}")
            with patch.object(builder, "SAVE_DIR", tmp), patch.object(
                builder,
                "download_model",
                return_value=(str(checkpoint), str(tokenizer)),
            ), patch.object(
                builder.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "convert"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    builder.build_trt_engine("small", config)
            self.assertFalse((Path(tmp) / "small" / config.identifier()).exists())
            self.assertEqual(list((Path(tmp) / "small").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
