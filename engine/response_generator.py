from __future__ import annotations

import logging
import os

import requests

from engine.prompt_builder import build_system_prompt, build_user_prompt
from models.schemas import ResponseContext


class BaseResponseGenerator:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.log_prompt = os.getenv("LLM_LOG_PROMPT", os.getenv("GLM_LOG_PROMPT", "true")).lower() == "true"
        self.logger = logging.getLogger("uvicorn.error")

    def _log_request(self, provider: str, system_prompt: str, user_prompt: str):
        if self.log_prompt:
            self.logger.info(
                "%s request prompt | model=%s | system_len=%d | user_len=%d\n"
                "----- SYSTEM PROMPT BEGIN -----\n%s\n"
                "----- SYSTEM PROMPT END -----\n"
                "----- USER PROMPT BEGIN -----\n%s\n"
                "----- USER PROMPT END -----",
                provider.upper(),
                self.model,
                len(system_prompt or ""),
                len(user_prompt or ""),
                system_prompt or "",
                user_prompt or "",
            )

    def _log_response(self, provider: str, content: str):
        if self.log_prompt:
            self.logger.info(
                "%s response | model=%s | len=%d\n"
                "----- %s RESPONSE BEGIN -----\n%s\n"
                "----- %s RESPONSE END -----",
                provider.upper(),
                self.model,
                len(content or ""),
                provider.upper(),
                content or "",
                provider.upper(),
            )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.85) -> str:
        raise NotImplementedError

    def generate(self, bible: dict, style: dict, ctx: ResponseContext) -> str:
        role_name = str(bible.get("name", "") or "").strip() or "角色"
        system_prompt = build_system_prompt(bible, style)
        user_prompt = build_user_prompt(ctx, role_name=role_name)
        content = self.chat(system_prompt, user_prompt, temperature=0.85)
        return content or "我在呢，怎么啦？"


class GLMResponseGenerator(BaseResponseGenerator):
    def __init__(self, api_key: str, model: str = "glm-4.7"):
        super().__init__(api_key=api_key, model=model)
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.85) -> str:
        self._log_request("glm", system_prompt, user_prompt)

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
        self._log_response("glm", content)
        return content


class OpenAIResponseGenerator(BaseResponseGenerator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key=api_key, model=model)
        self.url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.85) -> str:
        self._log_request("openai", system_prompt, user_prompt)

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
            "temperature": temperature,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.35,
        }
        resp = requests.post(self.url, headers=headers, json=data, timeout=150)
        if resp.status_code == 400:
            # Some models (e.g. gpt-5 family) reject tuning params such as temperature/top_p/penalties.
            retry_payload = {
                "model": self.model,
                "messages": data["messages"],
            }
            retry = requests.post(self.url, headers=headers, json=retry_payload, timeout=150)
            if retry.ok:
                body = retry.json()
                content = body["choices"][0]["message"]["content"].strip()
                self._log_response("openai", content)
                return content
            detail = (retry.text or "")[:1000]
            raise requests.HTTPError(f"OpenAI chat failed after fallback: {retry.status_code} {detail}", response=retry)

        if not resp.ok:
            detail = (resp.text or "")[:1000]
            raise requests.HTTPError(f"OpenAI chat failed: {resp.status_code} {detail}", response=resp)

        content = resp.json()["choices"][0]["message"]["content"].strip()
        self._log_response("openai", content)
        return content
