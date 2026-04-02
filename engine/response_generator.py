from __future__ import annotations

import logging
import os
import time

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
        self.max_retries = max(0, int(os.getenv("GLM_MAX_RETRIES", "5")))
        self.retry_backoff_sec = max(0.1, float(os.getenv("GLM_RETRY_BACKOFF_SEC", "15.0")))
        self.retry_max_backoff_sec = max(self.retry_backoff_sec, float(os.getenv("GLM_RETRY_MAX_BACKOFF_SEC", "60.0")))
        self.request_timeout_sec = max(5.0, float(os.getenv("GLM_REQUEST_TIMEOUT_SEC", "60.0")))
        self.busy_retry_wait_sec = max(
            self.retry_backoff_sec,
            float(os.getenv("GLM_BUSY_RETRY_WAIT_SEC", "60.0")),
        )
        self.fallback_model = str(os.getenv("GLM_FALLBACK_MODEL", "") or "").strip()

    @staticmethod
    def _is_busy_overload(status: int | None, detail: str) -> bool:
        if status != 429:
            return False
        lowered = (detail or "").lower()
        return "1305" in lowered or "访问量过大" in detail or "稍后再试" in detail

    def _sleep_before_retry(self, attempt: int, status: int | None = None, detail: str = "") -> None:
        wait_sec = min(self.retry_backoff_sec * (2 ** attempt), self.retry_max_backoff_sec)
        if self._is_busy_overload(status, detail):
            wait_sec = max(wait_sec, self.busy_retry_wait_sec)
            self.logger.warning(
                "GLM model busy, waiting %.1fs before retry | model=%s next_attempt=%d/%d",
                wait_sec,
                self.model,
                attempt + 2,
                self.max_retries + 1,
            )
        time.sleep(wait_sec)

    def _post_chat(self, model: str, headers: dict, data: dict) -> str:
        payload = dict(data)
        payload["model"] = model
        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.request_timeout_sec)
        if resp.ok:
            return resp.json()["choices"][0]["message"]["content"].strip()
        detail = (resp.text or "")[:1500]
        raise requests.HTTPError(f"GLM chat failed: {resp.status_code} {detail}", response=resp)

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
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content = self._post_chat(self.model, headers, data)
                self._log_response("glm", content)
                return content
            except requests.HTTPError as e:
                last_err = e
                status = (e.response.status_code if e.response is not None else None)
                detail = ((e.response.text if e.response is not None else str(e)) or "")[:600]
                retryable = status in {408, 409, 429, 500, 502, 503, 504}
                self.logger.error(
                    "GLM request failed | model=%s attempt=%d/%d status=%s detail=%s",
                    self.model,
                    attempt + 1,
                    self.max_retries + 1,
                    status,
                    detail,
                )
                if (not retryable) or attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt, status=status, detail=detail)
            except Exception as e:
                last_err = e
                self.logger.error(
                    "GLM request exception | model=%s attempt=%d/%d err=%s",
                    self.model,
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt)

        if self.fallback_model and self.fallback_model != self.model:
            self.logger.warning(
                "GLM primary model failed, retrying with fallback model: %s -> %s",
                self.model,
                self.fallback_model,
            )
            for attempt in range(self.max_retries + 1):
                try:
                    content = self._post_chat(self.fallback_model, headers, data)
                    self._log_response("glm", content)
                    return content
                except Exception as e:
                    last_err = e
                    self.logger.error(
                        "GLM fallback request failed | model=%s attempt=%d/%d err=%s",
                        self.fallback_model,
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                    )
                    if attempt >= self.max_retries:
                        break
                    if isinstance(e, requests.HTTPError):
                        status = (e.response.status_code if e.response is not None else None)
                        detail = ((e.response.text if e.response is not None else str(e)) or "")[:600]
                        self._sleep_before_retry(attempt, status=status, detail=detail)
                    else:
                        self._sleep_before_retry(attempt)

        if isinstance(last_err, Exception):
            raise last_err
        raise RuntimeError("GLM request failed without error context")


class OpenAIResponseGenerator(BaseResponseGenerator):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        thinking: dict | None = None,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ):
        super().__init__(api_key=api_key, model=model)
        root = str(base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
        self.url = root.rstrip("/") + "/chat/completions"
        self.thinking = thinking if isinstance(thinking, dict) else None
        self.reasoning_effort = str(reasoning_effort or "").strip() or None
        self.max_completion_tokens = min(int(max_completion_tokens), 131072) if max_completion_tokens else None

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
        if self.thinking:
            data["thinking"] = self.thinking
        if self.reasoning_effort:
            data["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            data["max_completion_tokens"] = self.max_completion_tokens
        self.logger.info(
            "OPENAI request config | model=%s thinking=%s reasoning_effort=%s max_completion_tokens=%s",
            self.model,
            bool(self.thinking),
            self.reasoning_effort or "",
            self.max_completion_tokens or 0,
        )
        resp = requests.post(self.url, headers=headers, json=data, timeout=150)
        if resp.status_code == 400:
            # Some models (e.g. gpt-5 family) reject tuning params such as temperature/top_p/penalties.
            retry_payload = {
                "model": self.model,
                "messages": data["messages"],
            }
            if self.thinking:
                retry_payload["thinking"] = self.thinking
            if self.reasoning_effort:
                retry_payload["reasoning_effort"] = self.reasoning_effort
            if self.max_completion_tokens:
                retry_payload["max_completion_tokens"] = self.max_completion_tokens
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
