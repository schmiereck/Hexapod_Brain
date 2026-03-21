#!/usr/bin/env python3
"""
Grog Bridge Node - LLM-based hexapod control using Groq API (OpenAI-compatible).
Model: meta-llama/llama-4-scout-17b-16e-instruct
"""

import base64
import io
import json
import os

import rclpy
from rclpy.executors import MultiThreadedExecutor

from openai import OpenAI
from PIL import Image as PILImage

from .base_bridge import BaseBridge
from .gemini_prompts import SYSTEM_INSTRUCTION


class GrogBridge(BaseBridge):
    def __init__(self):
        super().__init__('grog_bridge')

        # Grog-specific parameters
        self.declare_parameter('grog_api_key', '')
        self.declare_parameter('model_name', 'meta-llama/llama-4-scout-17b-16e-instruct')

        api_key = self.get_parameter('grog_api_key').value
        if not api_key:
            api_key = os.environ.get('GROG_API_KEY')
        if not api_key:
            self.get_logger().error('GROG_API_KEY not set! Set parameter or environment variable.')
            raise ValueError('GROG_API_KEY required')

        self.model_name = self.get_parameter('model_name').value

        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.groq.com/openai/v1',
        )

        self.get_logger().info(f'Grog client configured: {self.model_name}')

        self._start()

    def _call_llm(self, prompt: str, image_pil: PILImage.Image) -> dict:
        # Encode image as base64 JPEG for the OpenAI vision message format
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_INSTRUCTION},
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt + '\n\nRespond with valid JSON only, matching the required schema.',
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{img_b64}',
                            },
                        },
                    ],
                },
            ],
            response_format={'type': 'json_object'},
            timeout=self.timeout,
        )

        return json.loads(response.choices[0].message.content)


def main(args=None):
    rclpy.init(args=args)
    node = GrogBridge()
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
