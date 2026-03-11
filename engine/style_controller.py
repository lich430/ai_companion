from models.schemas import RelationshipState


class StyleController:
    def __init__(self, bible: dict):
        self.bible = bible

    def build_style_directives(self, state: RelationshipState) -> dict:
        reply_length = "short"
        if state.stage == "close" and state.mood in ["soft_support", "hurt", "recovering"]:
            reply_length = "medium"

        return {
            "reply_length": reply_length,
            "sentence_style": "口语短句",
            "allow_counter_question": state.initiative > 0.30,
            "allow_life_detail": state.disclosure > 0.20,
            "warmth_level": round(state.warmth, 2),
            "initiative_level": round(state.initiative, 2),
            "disclosure_level": round(state.disclosure, 2),
            "flirty_level": round(state.flirty, 2),
            "mood": state.mood,
            "stage": state.stage,
            "emoji_enabled": True,
            "marketing_allowed": False,
            "low_context": False,
            "clarify_needed": False,
        }
