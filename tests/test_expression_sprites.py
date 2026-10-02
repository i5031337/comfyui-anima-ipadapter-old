import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_expression_sprites as sprites


class ExpressionSpritesTests(unittest.TestCase):
    def build(self, **kwargs):
        return sprites.build_workflow("reference.png", "joy", 42, 12, 2.0, 0.7,
                                      "sprites/test", **kwargs)

    def run_main(self, *args):
        output = io.StringIO()
        with patch("sys.argv", ["generate_expression_sprites.py", *args]):
            with contextlib.redirect_stdout(output):
                sprites.main()
        return output.getvalue()

    def test_api_export_preserves_models_and_connections_with_new_ids(self):
        template = json.loads(Path(sprites.DEFAULT_WORKFLOW).read_text())
        template["26"] = {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {
            "unet_name": "my-model.gguf", "dequant_dtype": "float16",
            "patch_dtype": "float16", "patch_on_device": False,
        }}
        template["3"]["inputs"]["sampler_name"] = "heun"
        template["24"]["inputs"]["use_lora"] = True
        ids = {old: str(i + 100) for i, old in enumerate(template)}
        renamed = {}
        for old, node in template.items():
            for name, value in node["inputs"].items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in ids:
                    node["inputs"][name] = [ids[str(value[0])], value[1]]
            renamed[ids[old]] = node
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            path.write_text(json.dumps(renamed))
            result = self.build(workflow_path=path, width=640, height=960,
                                character="Test character", negative_prompt="bad art")
        for old in ("26", "23", "19", "20", "34:14", "34:13", "28"):
            self.assertEqual(result[ids[old]], renamed[ids[old]])
        sampler = result[ids["3"]]["inputs"]
        self.assertEqual(sampler["sampler_name"], "heun")
        self.assertEqual((sampler["seed"], sampler["steps"], sampler["cfg"]), (42, 12, 2.0))
        self.assertEqual(sampler["model"], renamed[ids["3"]]["inputs"]["model"])
        self.assertTrue(result[ids["24"]]["inputs"]["use_lora"])
        self.assertEqual(result[ids["24"]]["inputs"]["strength"], 0.7)
        self.assertIn("Test character", result[ids["6"]]["inputs"]["text"])
        self.assertEqual(result[ids["7"]]["inputs"]["text"], "bad art")
        self.assertEqual(result[ids["22"]]["inputs"]["image"], "reference.png")
        self.assertEqual(result[ids["5"]]["inputs"], {"width": 640, "height": 960, "batch_size": 1})
        self.assertEqual(result[ids["9"]]["inputs"]["filename_prefix"], "sprites/test")

    def test_editor_export_has_actionable_error(self):
        path = Path(sprites.DEFAULT_WORKFLOW).with_name("Anima_Turbo_IPAdapter.json")
        with self.assertRaisesRegex(ValueError, "API-format"):
            self.build(workflow_path=path)

    def test_lists_exit_without_loading_workflow_or_contacting_server(self):
        with tempfile.TemporaryDirectory() as directory:
            expressions = Path(directory) / "expressions.txt"
            expressions.write_text("'joy',\ncustom emotion\n")
            with patch.object(sprites, "queue_prompt") as queue:
                output = self.run_main("--list-styles", "--list-expressions", "--expressions",
                                       str(expressions), "--workflow", "missing.json", "--limit", "1")
                queue.assert_not_called()
        self.assertIn("retro, vintage, classic", output)
        self.assertIn("joy\ncustom emotion\n", output)

    def test_generation_requires_reference_before_creating_output_or_queueing(self):
        for args in ([], ["--ref-image", ""]):
            with self.subTest(args=args):
                error = io.StringIO()
                with patch.object(sprites.os, "makedirs") as mkdir:
                    with patch.object(sprites, "queue_prompt") as queue:
                        with contextlib.redirect_stderr(error):
                            with self.assertRaises(SystemExit) as raised:
                                self.run_main(*args)
                        mkdir.assert_not_called()
                        queue.assert_not_called()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--ref-image is required for generation", error.getvalue())

    def test_help_displays_relative_defaults_without_reference(self):
        output = io.StringIO()
        with patch("sys.argv", ["generate_expression_sprites.py", "--help"]):
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    sprites.main()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("./workflows/Expression_Sprites_API.json", output.getvalue())
        self.assertIn("./expressions.txt", output.getvalue())
        self.assertIn("../../output/expression_sprites", output.getvalue())

    def test_local_relative_and_absolute_images_upload_once_per_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            reference.write_bytes(b"reference image")
            expressions = root / "expressions.txt"
            expressions.write_text("joy\nanger\n")
            source = root / "generated.png"
            source.write_bytes(b"result")
            for image in (str(reference), os.path.relpath(reference)):
                with self.subTest(image=image):
                    with patch.object(sprites, "upload_reference_image", return_value="uploads/reference_1.png") as upload:
                        with patch.object(sprites, "queue_prompt", return_value={"prompt_id": "test"}) as queue:
                            with patch.object(sprites, "wait_for_prompt", return_value=(str(source), "")):
                                self.run_main("--ref-image", image, "--expressions", str(expressions),
                                              "--output-dir", directory)
                    upload.assert_called_once_with("127.0.0.1:8188", image)
                    self.assertEqual(queue.call_count, 2)
                    for call in queue.call_args_list:
                        self.assertEqual(call.args[1]["22"]["inputs"]["image"], "uploads/reference_1.png")

    def test_upload_uses_server_returned_name_and_subfolder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portrait.png"
            path.write_bytes(b"image bytes")
            response = io.BytesIO(json.dumps({"name": "reference (1).png", "subfolder": "portraits"}).encode())
            with patch.object(sprites.urllib.request, "urlopen", return_value=response) as urlopen:
                name = sprites.upload_reference_image("remote:8188", str(path))
            self.assertEqual(name, "portraits/reference (1).png")
            request = urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "http://remote:8188/upload/image")
            self.assertEqual(request.get_method(), "POST")
            self.assertIn(b'name="image"; filename="reference.png"', request.data)
            self.assertIn(b"image bytes", request.data)

    def test_server_input_filename_is_not_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generated.png"
            source.write_bytes(b"result")
            with patch.object(sprites, "upload_reference_image") as upload:
                with patch.object(sprites, "queue_prompt", return_value={"prompt_id": "test"}) as queue:
                    with patch.object(sprites, "wait_for_prompt", return_value=(str(source), "")):
                        self.run_main("--ref-image", "server_only/portrait.png", "--expression", "joy",
                                      "--output-dir", directory)
                upload.assert_not_called()
            self.assertEqual(queue.call_args.args[1]["22"]["inputs"]["image"], "server_only/portrait.png")

    def test_skip_existing_preserves_list_numbering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expressions = root / "expressions.txt"
            expressions.write_text("joy\nanger\n")
            existing = root / "sprite_01_joy.png"
            existing.write_bytes(b"existing")
            source = root / "generated.png"
            source.write_bytes(b"new")
            with patch.object(sprites, "queue_prompt", return_value={"prompt_id": "test"}) as queue:
                with patch.object(sprites, "wait_for_prompt", return_value=(str(source), "")):
                    output = self.run_main("--expressions", str(expressions), "--output-dir",
                                           directory, "--skip-existing", "--ref-image", "reference.png")
            self.assertEqual(queue.call_count, 1)
            self.assertIn("anger", queue.call_args.args[1]["6"]["inputs"]["text"])
            self.assertEqual(existing.read_bytes(), b"existing")
            self.assertEqual((root / "sprite_02_anger.png").read_bytes(), b"new")
            self.assertIn("Saved 1, skipped 1, failed to save 0", output)

    def test_skip_existing_single_expression_does_not_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "sprite_joy.png").write_bytes(b"existing")
            with patch.object(sprites, "queue_prompt") as queue:
                output = self.run_main("--expression", "joy", "--output-dir", directory,
                                       "--skip-existing", "--workflow", "missing.json",
                                       "--ref-image", "reference.png")
                queue.assert_not_called()
        self.assertIn("Saved 0, skipped 1, failed to save 0", output)

    def test_existing_single_expression_is_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "sprite_joy.png"
            destination.write_bytes(b"old")
            source = root / "generated.png"
            source.write_bytes(b"new")
            with patch.object(sprites, "queue_prompt", return_value={"prompt_id": "test"}) as queue:
                with patch.object(sprites, "wait_for_prompt", return_value=(str(source), "")):
                    self.run_main("--expression", "joy", "--output-dir", directory,
                                  "--ref-image", "reference.png")
            queue.assert_called_once()
            self.assertEqual(destination.read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
