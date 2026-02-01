# SPDX-License-Identifier: Apache-2.0
"""
Test case for update_weights_from_disk functionality in sglang-diffusion.

This test verifies that the DiffGenerator can update model weights from disk
without restarting the server, which is essential for RL workflows and
iterative fine-tuning.

Note: This test requires CUDA and cannot run on macOS/CPU-only environments.
"""

import unittest
import os
from typing import Optional


class TestDiffusionUpdateWeightsFromDisk(unittest.TestCase):
    """Test update_weights_from_disk for diffusion models."""

    # Use a small model for testing. For CI, consider using a smaller model.
    # The user requested "Use zai-org/GLM-Image as the test model".
    MODEL_PATH = "zai-org/GLM-Image"

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Skip test if CUDA is not available
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available, skipping diffusion tests")

        # Import here to avoid import errors on macOS
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        cls.DiffGenerator = DiffGenerator
        cls.ServerArgs = ServerArgs

    def setUp(self):
        """Set up test environment."""
        self.generator: Optional["DiffGenerator"] = None

    def tearDown(self):
        """Clean up after each test."""
        if self.generator is not None:
            try:
                self.generator.shutdown()
            except Exception:
                pass
            self.generator = None

    def _create_generator(self, model_path: str = None) -> "DiffGenerator":
        """Create a DiffGenerator instance."""
        model_path = model_path or self.MODEL_PATH
        server_args = self.ServerArgs(
            model_path=model_path,
            port=30001,  # Use a non-default port to avoid conflicts
        )
        return self.DiffGenerator.from_server_args(server_args, local_mode=True)

    def test_update_weights_method_exists(self):
        """Test that update_weights_from_disk method exists on DiffGenerator."""
        self.generator = self._create_generator()
        self.assertTrue(
            hasattr(self.generator, "update_weights_from_disk"),
            "DiffGenerator should have update_weights_from_disk method",
        )

    def test_update_weights_basic(self):
        """Test basic update_weights_from_disk functionality."""
        self.generator = self._create_generator()

        # Generate with initial weights
        prompt = "A simple test image"
        output_1 = self.generator.generate(
            {"prompt": prompt, "height": 256, "width": 256, "num_inference_steps": 2}
        )
        self.assertIsNotNone(output_1)

        # Update weights (reloading the same model to verify the mechanism works)
        result = self.generator.update_weights_from_disk(self.MODEL_PATH)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success", False), f"Update failed: {result}")

        # Generate with updated weights
        output_2 = self.generator.generate(
            {"prompt": prompt, "height": 256, "width": 256, "num_inference_steps": 2}
        )
        self.assertIsNotNone(output_2)

    def test_update_weights_invalid_path(self):
        """Test update_weights_from_disk with an invalid path."""
        self.generator = self._create_generator()

        # Try to update with a non-existent model path
        with self.assertRaises(RuntimeError):
            self.generator.update_weights_from_disk("/non/existent/model/path")


if __name__ == "__main__":
    unittest.main()
