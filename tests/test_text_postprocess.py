from engine.text_postprocess import postprocess_reply


class SeqRng:
    def __init__(self, values):
        self.values = list(values)

    def random(self):
        if self.values:
            return self.values.pop(0)
        return 0.99

    def choice(self, seq):
        return seq[0]

    def randint(self, a, b):
        return a


def test_postprocess_respects_emoji_disable():
    rng = SeqRng([0.20, 0.90, 0.90, 0.10])
    style = {"mood": "happy", "stage": "familiar", "emoji_enabled": False}
    out = postprocess_reply("你好。", style, rng)
    assert out == "你好"


def test_postprocess_replaces_common_punctuation_with_space():
    rng = SeqRng([0.10, 0.99, 0.99, 0.99])
    style = {"mood": "neutral", "stage": "familiar", "emoji_enabled": False}
    out = postprocess_reply("你来不来？我刚到，快点！", style, rng)
    assert "？" not in out and "！" not in out and "，" not in out
    assert " " in out


def test_postprocess_removes_modal_particles_strict():
    rng = SeqRng([0.99, 0.99, 0.99, 0.99])
    style = {"mood": "neutral", "stage": "familiar", "emoji_enabled": False}
    out = postprocess_reply("哎呀 你在干嘛呢", style, rng)
    assert "哎呀" not in out
    assert "呀" not in out
    assert "呢" not in out
