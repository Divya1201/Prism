from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

DEFAULT_MODEL_NAME = "google/flan-t5-base"


@lru_cache(maxsize=1)
def _build_generator(model_name: str = DEFAULT_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


class ExplainerService:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._generator = _build_generator(model_name)

    def generate_explanation(
        self,
        input_text: str,
        retrieved_evidence: Sequence[str] | Iterable[str],
        max_new_tokens: int = 128,
    ) -> dict[str, str]:
        evidence_lines = [f"- {line.strip()}" for line in retrieved_evidence if line and line.strip()]
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "- No evidence retrieved."

        prompt = (
            "Explain why this claim may be misleading using the following evidence:\n"
            f"Claim: {input_text.strip()}\n"
            f"Evidence:\n{evidence_block}"
        )

        result = self._generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        explanation = result[0]["generated_text"].strip()
        return {"explanation": explanation}
