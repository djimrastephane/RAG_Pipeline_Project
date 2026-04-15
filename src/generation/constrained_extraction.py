from __future__ import annotations

import json
from typing import Any, Optional


def build_constrained_extraction_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build strict extraction chat messages with fixed JSON output contract."""
    system = (
        "You are an exact extraction engine.\n"
        "You will only extract content from the provided CONTEXT.\n"
        "You will not paraphrase, summarize, reformat, or infer anything that you do not see "
        "character-for-character in the text.\n"
        "You must return EXACT JSON only, with no explanation, no comments, and no additional keys."
    )

    user = (
        "You MUST follow these rules:\n\n"
        "1. Use ONLY the provided CONTEXT.\n"
        "2. Find the exact answer text in the CONTEXT that answers the QUESTION.\n"
        "3. Copy the answer text VERBATIM — preserve spelling, punctuation, spaces, commas, "
        "parentheses, minus signs, and formatting.\n"
        "4. Do NOT reformat numbers or strip punctuation.\n"
        "5. Do NOT add symbols, units, or explanations.\n"
        "6. If there is more than one candidate that could be the answer, choose the one that exactly "
        "matches the context and best fits the question.\n"
        "7. If the gold answer does NOT appear exactly in the context, return null for both keys.\n"
        "8. The output MUST be valid JSON only. No other text.\n\n"
        f"QUESTION:\n{str(question)}\n\n"
        f"CONTEXT:\n{str(context)}\n\n"
        "OUTPUT JSON must be exactly in this format:\n"
        "{\n"
        '  "answer": "<exact copied substring from CONTEXT or null>",\n'
        '  "evidence_span": "<exact contiguous substring from CONTEXT that contains the answer or null>"\n'
        "}\n\n"
        "Return NOTHING other than this JSON object.\n"
        "END"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring, if any."""
    s = str(text or "")
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if start < 0:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def parse_constrained_extraction_response(text: str) -> dict[str, Optional[str]]:
    """
    Parse model output into strict schema:
    {"answer": str|None, "evidence_span": str|None}
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Empty response text; expected JSON object.")

    payload: Any = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        obj_text = _extract_first_json_object(raw)
        if not obj_text:
            raise ValueError("No JSON object found in model response.")
        try:
            payload = json.loads(obj_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON object in model response: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Parsed response is not a JSON object.")

    expected_keys = {"answer", "evidence_span"}
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            f"Invalid schema keys. Expected exactly {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )

    answer = payload.get("answer")
    evidence_span = payload.get("evidence_span")

    if answer is not None and not isinstance(answer, str):
        raise ValueError("Invalid type for 'answer'; expected string or null.")
    if evidence_span is not None and not isinstance(evidence_span, str):
        raise ValueError("Invalid type for 'evidence_span'; expected string or null.")

    return {
        "answer": answer,
        "evidence_span": evidence_span,
    }


def validate_span_in_context(
    answer: str | None,
    evidence_span: str | None,
    context: str,
) -> dict[str, Any]:
    """Validate that extracted spans are verbatim substrings of context."""
    ctx = str(context or "")
    out_answer = answer
    out_evidence = evidence_span
    violations: list[str] = []

    if out_answer is not None and out_answer not in ctx:
        violations.append("answer_not_in_context")
        out_answer = None
        out_evidence = None
        return {
            "answer": out_answer,
            "evidence_span": out_evidence,
            "violations": violations,
        }

    if out_evidence is not None and out_evidence not in ctx:
        violations.append("evidence_span_not_in_context")
        out_answer = None
        out_evidence = None

    return {
        "answer": out_answer,
        "evidence_span": out_evidence,
        "violations": violations,
    }


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Fallback prompt serialization for non-chat local clients."""
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user").upper()
        content = str(m.get("content") or "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _extract_chat_content(response: Any) -> str:
    """Extract assistant text from chat completion response shape."""
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text") or ""))
                    elif hasattr(item, "text"):
                        texts.append(str(getattr(item, "text") or ""))
                return "".join(texts)
    if isinstance(response, dict):
        try:
            return str(response["choices"][0]["message"]["content"])
        except Exception:
            return ""
    return ""


def run_constrained_extraction(
    llm_client: Any,
    question: str,
    context: str,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 200,
) -> dict[str, Any]:
    """Run strict constrained extraction through chat API or repo-local generate API."""
    messages = build_constrained_extraction_messages(question=question, context=context)

    raw_text = ""

    chat_api = getattr(getattr(llm_client, "chat", None), "completions", None)
    create_fn = getattr(chat_api, "create", None)
    if callable(create_fn):
        model = getattr(llm_client, "model", None)
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if model:
            kwargs["model"] = model
        response = create_fn(**kwargs)
        raw_text = _extract_chat_content(response)
    else:
        generate_fn = getattr(llm_client, "generate", None)
        if not callable(generate_fn):
            raise ValueError("llm_client must expose chat.completions.create(...) or generate(prompt).")
        prompt = _messages_to_prompt(messages)
        try:
            generated = generate_fn(
                prompt,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
        except TypeError:
            generated = generate_fn(prompt)
        raw_text = str(getattr(generated, "answer", generated) or "")

    parsed = parse_constrained_extraction_response(raw_text)
    validated = validate_span_in_context(
        answer=parsed.get("answer"),
        evidence_span=parsed.get("evidence_span"),
        context=context,
    )
    return validated
