from dataclasses import asdict, dataclass
import math

from .io import digest, read_json

DATASETS = ("iacv1", "iacv2", "semeval", "mustard")
PAPER_COUNTS = {"iacv1": (1595, 320), "iacv2": (5216, 1042), "semeval": (3634, 784), "mustard": (552, 138)}


@dataclass(frozen=True)
class Config:
    profile: str = "appendix"
    model: str = "gpt-4o"
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 512
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    k: int = 3
    retrieval_order_seed: int = 0
    use_context_agent: bool = True
    simple_irony_semantic_support: bool = True
    planner_rule_validation: bool = True
    output_token_parameter: str = "max_tokens"
    reasoning_effort: str | None = None

    def __post_init__(self):
        if self.profile not in ("appendix", "maintext"):
            raise ValueError("profile must be appendix or maintext")
        if self.output_token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("Unknown output token parameter")
        if self.reasoning_effort not in (None, "none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unknown reasoning effort")
        for name in ("max_tokens", "embedding_dimensions", "k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.retrieval_order_seed) is not int or type(self.n) is not int or self.n != 1:
            raise ValueError("Use an integer retrieval_order_seed and n=1")
        for name in ("use_context_agent", "simple_irony_semantic_support", "planner_rule_validation"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name, lo, hi in (("temperature", 0, 2), ("top_p", 0, 1), ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {name}")
        if not isinstance(self.model, str) or not isinstance(self.embedding_model, str) or not self.model.strip() or not self.embedding_model.strip():
            raise ValueError("Model names cannot be empty")

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))

    def as_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.as_dict())

    def generation(self):
        payload = {name: getattr(self, name) for name in ("model", "temperature", "top_p", "n", "frequency_penalty", "presence_penalty")}
        payload[self.output_token_parameter] = self.max_tokens
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def retrieves(self, dataset):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        return dataset != "mustard"
