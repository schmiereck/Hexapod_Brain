#!/usr/bin/env python3
"""
Gemini Bridge Node - LLM-based hexapod control using Google Gemini API.
"""

import json
import os

import rclpy
from rclpy.executors import MultiThreadedExecutor

from google import genai
from PIL import Image as PILImage

from .base_bridge import BaseBridge
from .gemini_prompts import SYSTEM_INSTRUCTION, RESPONSE_SCHEMA


class GeminiBridge(BaseBridge):
    def __init__(self):
        super().__init__('gemini_bridge')

        # Gemini-specific parameters
        self.declare_parameter('gemini_api_key', '')
        self.declare_parameter('model_name', 'models/gemini-robotics-er-1.5-preview')

        api_key = self.get_parameter('gemini_api_key').value
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY not set! Set parameter or environment variable.')
            raise ValueError('GEMINI_API_KEY required')

        self.model_name = self.get_parameter('model_name').value

        self.client = genai.Client(api_key=api_key)
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "system_instruction": SYSTEM_INSTRUCTION,
        }

        mode_str = "YOLO+Gemini" if self.use_yolo else "Pure Vision"
        self.get_logger().info(f'Gemini client configured: {self.model_name} - Mode: {mode_str}')

        self._start()

    def _call_llm(self, prompt: str, image_pil: PILImage.Image) -> dict:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image_pil],
            config=self.generation_config
        )
        return json.loads(response.text)


def main(args=None):
    rclpy.init(args=args)
    node = GeminiBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
