from __future__ import annotations

import logging
import os

import requests

from engine.prompt_builder import build_system_prompt, build_user_prompt
from models.schemas import ResponseContext


class GLMResponseGenerator:
    def __init__(self, api_key: str, model: str = "glm-4.7"):
        self.api_key = api_key
        self.model = model
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.log_prompt = os.getenv("GLM_LOG_PROMPT", "true").lower() == "true"
        self.logger = logging.getLogger("uvicorn.error")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.85) -> str:
        if self.log_prompt:
            self.logger.info(
                "GLM request prompt | model=%s | system_len=%d | user_len=%d\n"
                "----- SYSTEM PROMPT BEGIN -----\n%s\n"
                "----- SYSTEM PROMPT END -----\n"
                "----- USER PROMPT BEGIN -----\n%s\n"
                "----- USER PROMPT END -----",
                self.model,
                len(system_prompt or ""),
                len(user_prompt or ""),
                system_prompt or "",
                user_prompt or "",
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.35,
        }
        resp = requests.post(self.url, headers=headers, json=data, timeout=150)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if self.log_prompt:
            self.logger.info(
                "GLM response | model=%s | len=%d\n"
                "----- GLM RESPONSE BEGIN -----\n%s\n"
                "----- GLM RESPONSE END -----",
                self.model,
                len(content or ""),
                content or "",
            )
        return content

    def generate(self, bible: dict, style: dict, ctx: ResponseContext) -> str:
        role_name = str(bible.get("name", "") or "").strip() or "角色"
        system_prompt = build_system_prompt(bible, style)
        user_prompt = build_user_prompt(ctx, role_name=role_name)
        content = self.chat(system_prompt, user_prompt, temperature=0.85)
        return content or "我在呢，怎么啦？"

