"""WorldGym scoring with an injected judge client and response records."""
import re
from types import FunctionType, SimpleNamespace

from wm_repr.third_party.worldgym import utils as upstream

encode_video = upstream.encode_video
aggregate_model_results = upstream.aggregate_model_results
print_results_table = upstream.print_results_table


def predict(video, trial, n=3, return_raw=False, *, client):
    """Run the upstream scorer and optionally retain each judge response.

    Each call has its own function namespace so concurrent judges can use
    different clients without changing the upstream module's globals.
    """
    responses = []

    def create(**kwargs):
        response = client.chat.completions.create(**kwargs)
        responses.append(response)
        return response

    proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    namespace = dict(upstream.predict.__globals__)
    namespace["OpenAI"] = lambda *args, **kwargs: proxy
    scorer = FunctionType(upstream.predict.__code__, namespace,
                          upstream.predict.__name__, upstream.predict.__defaults__,
                          upstream.predict.__closure__)
    score = scorer(video, trial, n=n)
    if not return_raw:
        return score

    # Parsing here records individual replies; upstream determines the final vote.
    allowed = r"(0\.5|0(?:\.0)?|1(?:\.0)?)" if trial.get("partial_criteria") else r"(0(?:\.0)?|1(?:\.0)?)"
    pattern = re.compile(rf"Final\s*Score:\s*{allowed}", re.IGNORECASE)
    per_call = []
    for response in responses:
        usage = getattr(response, "usage", None)
        for choice in response.choices:
            content = (choice.message.content or "").strip()
            match = pattern.search(content)
            per_call.append({
                "parsed_score": float(match.group(1)) if match else None,
                "rationale": content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "model": getattr(response, "model", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "response_id": getattr(response, "id", None),
            })
    return score, per_call
