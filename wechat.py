import base64
from difflib import SequenceMatcher
import json
import os
import re
import subprocess
import time

import requests
import uiautomator2 as u2
from dotenv import load_dotenv
from loguru import logger

# ===================== config =====================
load_dotenv()
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
ADB_PATH = os.getenv("ADB_PATH", "adb")
PHONE_IP = os.getenv("PHONE_IP", "")  # Optional: if empty, auto-detect first connected device


def get_first_adb_device(adb_path: str = "adb") -> str | None:
    """Scan for connected ADB devices and return the first one."""
    try:
        result = subprocess.run(
            [adb_path, "devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:  # Skip header "List of devices attached"
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                device_id = parts[0]
                logger.info(f"Found ADB device: {device_id}")
                return device_id
        logger.warning("No connected ADB device found")
        return None
    except Exception as e:
        logger.error(f"Failed to scan ADB devices: {e}")
        return None


def check_device_connected(expected_device: str, adb_path: str = "adb") -> bool:
    """Check if the expected device is still connected."""
    try:
        result = subprocess.run(
            [adb_path, "-s", expected_device, "get-state"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
        return result.returncode == 0 and "device" in result.stdout.strip()
    except Exception:
        return False
MAX_VISION_RETRY = 3
LOOP_INTERVAL = 10
ENGINE_API_BASE = os.getenv("ENGINE_API_BASE", "http://45.119.97.94:8080/")
ENGINE_API_TIMEOUT = int(os.getenv("ENGINE_API_TIMEOUT", "150"))
CHAT_ID_MATCH_THRESHOLD = float(os.getenv("CHAT_ID_MATCH_THRESHOLD", "0.62"))

LOGIC_CANVAS_WIDTH = 1000
LOGIC_CANVAS_HEIGHT = 1000
Y_OFFSET = -60

CHAT_DEFAULT_INPUT_BOX = [400, 950]
CHAT_DEFAULT_SEND_BTN = [900, 950]
DEFAULT_WECHAT_TAB = [135, 950]
LIST_PAGE_UNREAD_DEFAULT_COORDS = [500, 300]
EXCLUDED_SESSION_KEYWORDS = [
    "公众号",
    "服务号",
    "腾讯新闻",
    "微信支付",
    "微信游戏",
    "订阅号",
    "视频号",
    "文件传输助手",
    "腾讯客服",
    "腾讯充值",
    "微信团队",
    "微信运动",
    "小程序",
    "群聊",
    "助手",
]
LIST_SAFE_X = int(os.getenv("LIST_SAFE_X", "500"))
PENDING_PULL_LIMIT = int(os.getenv("PENDING_PULL_LIMIT", "5"))
OPEN_CHAT_ACTION = os.getenv("OPEN_CHAT_ACTION", "com.example.wechatglm.OPEN_CHAT")
OPEN_CHAT_EXTRA_KEY = os.getenv("OPEN_CHAT_EXTRA_KEY", "wxid")
CHAT_PAGE_CACHE_TTL_SEC = float(os.getenv("CHAT_PAGE_CACHE_TTL_SEC", "6"))
SEND_CHUNK_DELAY_MS_MIN = int(os.getenv("SEND_CHUNK_DELAY_MS_MIN", "350"))
SEND_CHUNK_DELAY_MS_MAX = int(os.getenv("SEND_CHUNK_DELAY_MS_MAX", "900"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "20"))
CHAT_DEFAULT_EMOJI_BTN = [780, 950]
CHAT_DEFAULT_EMOJI_CUSTOM_TAB = [110, 880]
EMOJI_BTN_COORDS_RAW = os.getenv("EMOJI_BTN_COORDS", "780,950")
EMOJI_CUSTOM_TAB_COORDS_RAW = os.getenv("EMOJI_CUSTOM_TAB_COORDS", "110,880")
EMOJI_SLOTS_RAW = os.getenv(
    "EMOJI_SLOTS",
    "120,780;260,780;400,780;540,780;680,780;820,780",
)
EMOJI_CMD_PATTERN = re.compile(r"^\[(?:WX_)?EMOJI:(\d+)\]$", re.IGNORECASE)
_chat_page_cache = {"ts": 0.0, "data": None}

# Backend API should not inherit system proxy settings, otherwise local/remote
# service calls may be redirected to a dead proxy like 127.0.0.1:7897.
backend_session = requests.Session()
backend_session.trust_env = False


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def _is_same_session(target_name: str, chat_id: str, threshold: float = CHAT_ID_MATCH_THRESHOLD) -> tuple[bool, float]:
    score = _text_similarity(target_name, chat_id)
    return score >= threshold, score


def _sanitize_reply_particles(text: str) -> str:
    out = (text or "").strip()
    if not out:
        return out
    out = out.replace("哎呀", " ")
    out = out.replace("呀", " ")
    out = out.replace("呢", " ")
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def _extract_reply_parts(reply: str) -> list[str]:
    raw = (reply or "").strip()
    if not raw:
        return []

    # Remove markdown fence if model returns ```json ... ```
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    candidates = [fenced, raw]

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                continue
            try:
                data = json.loads(match.group(0))
            except Exception:
                continue

        if isinstance(data, dict):
            if isinstance(data.get("replies"), list):
                return [str(x).strip() for x in data["replies"] if str(x).strip()]
            if isinstance(data.get("reply"), str) and data.get("reply", "").strip():
                return [data["reply"].strip()]

    return [x.strip() for x in re.split(r"\n+", raw) if x.strip()]


def _split_chunks_semantic(text: str, max_chars: int) -> list[str]:
    max_chars = max(1, int(max_chars))
    lines = [line.strip() for line in re.split(r"\n+", text or "") if line.strip()]
    out: list[str] = []
    for line in lines:
        words = [word for word in re.split(r"\s+", line) if word]
        if not words:
            continue

        current = ""
        for word in words:
            if len(word) > max_chars:
                if current:
                    out.append(current)
                    current = ""
                start = 0
                while start < len(word):
                    out.append(word[start : start + max_chars].strip())
                    start += max_chars
                continue

            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    out.append(current)
                current = word

        if current:
            out.append(current)
    return [x for x in out if x]


def _parse_coord_pair(raw: str) -> list[int] | None:
    text = (raw or "").strip()
    if not text:
        return None
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        return None
    try:
        return [int(parts[0]), int(parts[1])]
    except Exception:
        return None


def _load_emoji_slots() -> list[list[int]]:
    slots: list[list[int]] = []
    for token in (EMOJI_SLOTS_RAW or "").split(";"):
        parsed = _parse_coord_pair(token)
        if parsed:
            slots.append(parsed)
    return slots


EMOJI_SLOTS = _load_emoji_slots()
EMOJI_BTN_COORDS = _parse_coord_pair(EMOJI_BTN_COORDS_RAW) or CHAT_DEFAULT_EMOJI_BTN
EMOJI_CUSTOM_TAB_COORDS = _parse_coord_pair(EMOJI_CUSTOM_TAB_COORDS_RAW) or CHAT_DEFAULT_EMOJI_CUSTOM_TAB


def _parse_emoji_command(part: str) -> int | None:
    text = (part or "").strip()
    if not text:
        return None
    m = EMOJI_CMD_PATTERN.match(text)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if idx >= 1 else None


class ScreenCoordinateConverter:
    def __init__(self, adb_path, phone_ip, logic_width, logic_height, y_offset=0):
        self.adb_path = adb_path
        self.phone_ip = phone_ip
        self.logic_width = logic_width
        self.logic_height = logic_height
        self.y_offset = y_offset
        self.screen_width = 0
        self.screen_height = 0
        self.status_bar_height = 0
        self.nav_bar_height = 0
        self._get_real_screen_params()

    def _run_adb_cmd(self, cmd_list, timeout=5):
        full_cmd = [self.adb_path, "-s", self.phone_ip] + cmd_list
        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout
            )
            return result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            logger.error(f"ADB command failed: {' '.join(full_cmd)} error={e}")
            return "", str(e)

    def _get_real_screen_params(self):
        width_output, _ = self._run_adb_cmd(["shell", "wm", "size"])
        if "Physical size:" in width_output:
            res_str = width_output.split("Physical size:")[1].strip()
            self.screen_width, self.screen_height = map(int, res_str.split("x"))

        # optional values, may be empty depending on device
        status_output, _ = self._run_adb_cmd(["shell", "dumpsys", "window"])
        status_match = re.search(r"status_bar_height=(\d+)", status_output)
        if status_match:
            self.status_bar_height = int(status_match.group(1))

        nav_match = re.search(r"navigation_bar_height=(\d+)", status_output)
        if nav_match:
            self.nav_bar_height = int(nav_match.group(1))

        logger.info(
            f"Screen params: logic={self.logic_width}x{self.logic_height}, "
            f"physical={self.screen_width}x{self.screen_height}, "
            f"status={self.status_bar_height}, nav={self.nav_bar_height}, y_offset={self.y_offset}"
        )

    def convert(self, logic_x, logic_y):
        if not isinstance(logic_x, (int, float)) or not isinstance(logic_y, (int, float)):
            logger.error(f"Invalid logic coordinate: ({logic_x},{logic_y})")
            return 0, 0

        available_height = self.screen_height - self.status_bar_height - self.nav_bar_height
        scale_x = self.screen_width / self.logic_width
        scale_y = available_height / self.logic_height

        scaled_x = logic_x * scale_x
        scaled_y = logic_y * scale_y

        physical_y = scaled_y + self.status_bar_height + self.y_offset
        physical_x = max(0, min(int(scaled_x), self.screen_width - 1))
        physical_y = max(0, min(int(physical_y), self.screen_height - 1))
        return physical_x, physical_y


# Auto-detect first ADB device if PHONE_IP not set
if not PHONE_IP:
    PHONE_IP = get_first_adb_device(ADB_PATH)
    if not PHONE_IP:
        logger.error("No ADB device found and PHONE_IP not set")
        raise SystemExit(1)

coord_converter = ScreenCoordinateConverter(
    adb_path=ADB_PATH,
    phone_ip=PHONE_IP,
    logic_width=LOGIC_CANVAS_WIDTH,
    logic_height=LOGIC_CANVAS_HEIGHT,
    y_offset=Y_OFFSET,
)

try:
    d = u2.connect(PHONE_IP)
    d.set_input_ime(True)
    logger.info(f"uiautomator2 connected: {PHONE_IP}")
except Exception as e:
    logger.error(f"uiautomator2 connect failed: {e}")
    logger.error("Run: python -m uiautomator2 init")
    raise SystemExit(1)


def run_adb_cmd(cmd_list, timeout=10):
    full_cmd = [ADB_PATH, "-s", PHONE_IP] + cmd_list
    result = subprocess.run(
        full_cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout
    )
    if result.returncode != 0:
        logger.warning(f"ADB non-zero: {' '.join(full_cmd)} err={result.stderr}")
    return result


def ensure_wechat_list_page():
    try:
        current_app = d.app_current()
        if current_app and current_app.get("package") == "com.tencent.mm":
            tab_x, tab_y = coord_converter.convert(*DEFAULT_WECHAT_TAB)
            d.click(tab_x, tab_y)
            time.sleep(1)
            return True

        d.app_start("com.tencent.mm", stop=False)
        time.sleep(3)
        tab_x, tab_y = coord_converter.convert(*DEFAULT_WECHAT_TAB)
        d.click(tab_x, tab_y)
        time.sleep(1)
        return True
    except Exception as e:
        logger.error(f"Enter list page failed: {e}")
        return False

def adb_screenshot():
    screenshot_path = "/sdcard/wechat_auto_screenshot.png"
    run_adb_cmd(["shell", "rm", "-f", screenshot_path], timeout=3)
    run_adb_cmd(["shell", "screencap", "-p", screenshot_path], timeout=5)
    run_adb_cmd(["pull", screenshot_path, "./temp_screenshot.png"], timeout=5)
    with open("./temp_screenshot.png", "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def uiautomator2_operate(operate_type, params):
    try:
        if operate_type == "click":
            if not isinstance(params, list) or len(params) != 2:
                logger.error(f"invalid click params: {params}")
                return False
            logic_x, logic_y = params
            x, y = coord_converter.convert(logic_x, logic_y)
            d.click(x, y)

        elif operate_type == "input":
            text = params.strip() if isinstance(params, str) else ""
            if not text:
                text = "我看到你的消息了，刚在忙。"
            d.send_keys(text)

        time.sleep(1)
        return True
    except Exception as e:
        logger.error(f"uiautomator2 operate failed ({operate_type}): {e}")
        return False


def send_custom_emoji(slot_index: int) -> bool:
    if slot_index <= 0:
        logger.error("invalid emoji slot index: {}", slot_index)
        return False
    if not EMOJI_SLOTS:
        logger.error("EMOJI_SLOTS is empty; cannot send custom emoji")
        return False
    if slot_index > len(EMOJI_SLOTS):
        logger.error("emoji slot out of range: idx={} max={}", slot_index, len(EMOJI_SLOTS))
        return False

    emoji_btn = EMOJI_BTN_COORDS
    custom_tab = EMOJI_CUSTOM_TAB_COORDS
    target_slot = EMOJI_SLOTS[slot_index - 1]

    logger.info(
        "sending custom emoji: slot={} btn={} tab={} target={}",
        slot_index,
        emoji_btn,
        custom_tab,
        target_slot,
    )

    if not uiautomator2_operate("click", emoji_btn):
        logger.error("click emoji panel button failed")
        return False
    time.sleep(0.4)

    if not uiautomator2_operate("click", custom_tab):
        logger.error("click emoji custom tab failed")
        return False
    time.sleep(0.4)

    if not uiautomator2_operate("click", target_slot):
        logger.error("click emoji target slot failed")
        return False

    time.sleep(0.4)
    return True

def call_glm_4v(img_base64, prompt):
    logger.info(
        "glm-4.6v prompt | len={}\n----- VISION PROMPT BEGIN -----\n{}\n----- VISION PROMPT END -----",
        len(prompt or ""),
        prompt or "",
    )
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Authorization": f"Bearer {GLM_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "glm-4.6v",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            ],
        }],
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=150)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"call glm-4v failed: {e}")
        return "{}"

def fetch_pending_replies(limit=PENDING_PULL_LIMIT, pop=True):
    try:
        url = f"{ENGINE_API_BASE.rstrip('/')}/hook/pending"
        params = {"limit": int(limit), "pop": str(bool(pop)).lower()}
        logger.info(f"pending api request: url={url}, params={params}")
        resp = backend_session.get(url, params=params, timeout=ENGINE_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        if not isinstance(items, list):
            return []
        return items
    except Exception as e:
        logger.error(f"fetch pending replies failed: {e}")
        return []

def open_chat_via_broadcast(username: str) -> bool:
    wxid = (username or "").strip()
    if not wxid:
        return False
    try:
        cmd = [
            "shell",
            "am",
            "broadcast",
            "-a",
            OPEN_CHAT_ACTION,
            "--es",
            OPEN_CHAT_EXTRA_KEY,
            wxid,
        ]
        result = run_adb_cmd(cmd, timeout=10)
        logger.info(
            f"open_chat_via_broadcast: action={OPEN_CHAT_ACTION}, {OPEN_CHAT_EXTRA_KEY}={wxid}, "
            f"stdout={result.stdout.strip()}"
        )
        time.sleep(1.2)
        return True
    except Exception as e:
        logger.error(f"open_chat_via_broadcast failed: {e}")
        return False


def send_reply_in_chat(reply: str, recommended_delay_ms: int = 0) -> bool:
    logger.info("wechat send raw reply: {!r}", reply)
    parts = _extract_reply_parts(reply)
    logger.info("wechat parsed reply parts({}): {}", len(parts), parts)
    chunks = [_sanitize_reply_particles(x) for x in parts if _sanitize_reply_particles(x)]
    logger.info("wechat sanitized reply chunks({}): {}", len(chunks), chunks)
    if not chunks:
        return False

    chat_page_data = recognize_chat_page()
    logger.info(f"chat_page_data:{chat_page_data}")

    if chat_page_data.get("page_type") not in {"list", "chat"}:
        return False
    if chat_page_data.get("page_type") != "chat":
        logger.warning(f"expected chat page, got {chat_page_data.get('page_type')}")
        return False

    if not uiautomator2_operate("click", chat_page_data["input_box_coords"]):
        logger.error("click input box failed")
        return False
    time.sleep(1)

    if recommended_delay_ms > 0:
        wait_s = max(0.0, recommended_delay_ms / 1000.0)
        logger.info(f"wait recommended delay before send: {wait_s:.2f}s")
        time.sleep(wait_s)

    for part in chunks:
        if not part:
            continue
        logger.info("wechat sending chunk: {!r}", part)
        emoji_slot = _parse_emoji_command(part)

        part_delay_ms = max(
            SEND_CHUNK_DELAY_MS_MIN,
            min(SEND_CHUNK_DELAY_MS_MAX, recommended_delay_ms // 2 if recommended_delay_ms else SEND_CHUNK_DELAY_MS_MIN),
        )
        time.sleep(part_delay_ms / 1000.0)

        if emoji_slot is not None:
            if not send_custom_emoji(emoji_slot):
                logger.error("send custom emoji failed, slot={}", emoji_slot)
                return False
            continue

        if not uiautomator2_operate("input", part):
            logger.error("input message failed")
            return False
        if not uiautomator2_operate("click", chat_page_data["send_btn_coords"]):
            logger.error("click send button failed")
            return False
        time.sleep(0.3)
    return True


def process_pending_replies():
    logger.info("=" * 50)
    logger.info("start processing pending replies")
    logger.info("=" * 50)

    if not ensure_wechat_list_page():
        logger.error("failed to enter wechat list page")
        return False

    items = fetch_pending_replies(limit=PENDING_PULL_LIMIT, pop=True)
    if not items:
        logger.info("no pending replies")
        return False

    handled = 0
    for item in items:
        username = str(item.get("username", "")).strip()
        reply = str(item.get("reply", "")).strip()
        recommended_delay_ms = int(item.get("recommended_delay_ms", 0) or 0)
        if not username or not reply:
            logger.warning(f"skip invalid pending item: {item}")
            continue
        logger.info(
            f"process pending item: username={username}, reply_len={len(reply)}, "
            f"recommended_delay_ms={recommended_delay_ms}"
        )
        if not open_chat_via_broadcast(username):
            logger.error(f"open chat via broadcast failed: {username}")
            continue
        if not send_reply_in_chat(reply, recommended_delay_ms=recommended_delay_ms):
            logger.error(f"send reply failed: {username}")
            d.press("back")
            time.sleep(0.5)
            continue
        handled += 1
        d.press("back")
        time.sleep(0.8)
        tab_x, tab_y = coord_converter.convert(*DEFAULT_WECHAT_TAB)
        d.click(tab_x, tab_y)
        time.sleep(0.8)

    logger.info(f"pending replies handled: {handled}/{len(items)}")
    return handled > 0

def get_chat_page_prompt():
    return f"""
你是微信聊天页面视觉识别助手，仅返回JSON格式结果，无任何其他文字。
识别规则：
0) 先判断页面类型，若不是聊天页返回 page_type=unknown。
1) 聊天页特征：顶部有用户名，中间消息列表，底部有输入框、笑脸、加号。
2) 返回输入框和发送按钮坐标（1000x1000逻辑坐标）。
   默认：input={CHAT_DEFAULT_INPUT_BOX}, send={CHAT_DEFAULT_SEND_BTN}

输出字段：
{{
  "page_type": "chat|list|unknown",
  "input_box_coords": [x, y],
  "send_btn_coords": [x, y]
}}
"""


def validate_chat_page_result(vision_result):
    try:
        json_str = re.search(r"\{.*\}", vision_result, re.DOTALL).group()
        vision_data = json.loads(json_str)
    except Exception as e:
        logger.error(f"chat page parse failed: {e}")
        vision_data = {
            "page_type": "unknown",
            "input_box_coords": [],
            "send_btn_coords": [],
        }

    vision_data["page_type"] = vision_data.get("page_type", "unknown")
    vision_data["input_box_coords"] = vision_data.get("input_box_coords", [])
    vision_data["send_btn_coords"] = vision_data.get("send_btn_coords", [])

    if not isinstance(vision_data["input_box_coords"], list) or len(vision_data["input_box_coords"]) != 2:
        vision_data["input_box_coords"] = []
    if not isinstance(vision_data["send_btn_coords"], list) or len(vision_data["send_btn_coords"]) != 2:
        vision_data["send_btn_coords"] = []

    return vision_data


def recognize_chat_page(use_cache=True):
    now = time.time()
    if use_cache:
        cached = _chat_page_cache.get("data")
        cached_ts = float(_chat_page_cache.get("ts", 0.0) or 0.0)
        if cached and (now - cached_ts) <= CHAT_PAGE_CACHE_TTL_SEC:
            return cached

    prompt = get_chat_page_prompt()
    for retry in range(MAX_VISION_RETRY):
        try:
            img_base64 = adb_screenshot()
            vision_result = call_glm_4v(img_base64, prompt)
            data = validate_chat_page_result(vision_result)
            _chat_page_cache["ts"] = time.time()
            _chat_page_cache["data"] = data
            return data
        except Exception as e:
            logger.warning(f"chat page recognize retry={retry + 1} err={e}")
            time.sleep(0.5)

    data = {
        "page_type": "unknown",
        "input_box_coords": [],
        "send_btn_coords": [],
    }
    _chat_page_cache["ts"] = time.time()
    _chat_page_cache["data"] = data
    return data

if __name__ == "__main__":
    logger.add("wechat_auto_reply.log", rotation="100MB", level="INFO", encoding="utf-8")

    if not PHONE_IP:
        logger.error("Missing PHONE_IP")
        raise SystemExit(1)

    DEVICE_CHECK_INTERVAL = 30  # Check device connection every 30 seconds
    last_device_check = 0

    logger.info(f"auto loop started, interval={LOOP_INTERVAL}s")
    while True:
        try:
            # Periodically check if device is still connected
            now = time.time()
            if now - last_device_check > DEVICE_CHECK_INTERVAL:
                if not check_device_connected(PHONE_IP, ADB_PATH):
                    logger.error(f"Device {PHONE_IP} disconnected, exiting for restart...")
                    raise SystemExit(1)
                last_device_check = now

            process_pending_replies()
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"main loop error: {e}")
        time.sleep(LOOP_INTERVAL)
