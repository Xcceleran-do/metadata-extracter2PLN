from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    )


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return (
        default
        if value is None
        else value.strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class Settings:
    environment: str = "production"
    api_keys: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    model_provider: str = "bedrock"
    bedrock_model: str = ""
    bedrock_region: str = "us-east-1"
    bedrock_access_key: str | None = None
    bedrock_secret_key: str | None = None
    bedrock_max_tokens: int = 8192
    jev_model: str = "jev-latest"
    jev_api_key: str | None = None
    jev_transport: str = "typesafe"
    openrouter_api_key: str | None = None
    openrouter_model: str = "typesafe/jev-1.13"
    model_timeout_seconds: float = 45.0
    request_timeout_seconds: float = 120.0
    max_request_bytes: int = 2_097_152
    max_concurrent_requests: int = 4
    rate_limit_per_minute: int = 60
    expose_docs: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.getenv("METADATA_ENVIRONMENT", "production"),
            api_keys=_csv("METADATA_API_KEYS"),
            allowed_hosts=_csv("METADATA_ALLOWED_HOSTS", "localhost,127.0.0.1"),
            bedrock_model=os.getenv("METADATA_BEDROCK_MODEL_ID", "").strip(),
            bedrock_region=os.getenv("AWS_REGION", "us-east-1").strip(),
            bedrock_access_key=os.getenv("AWS_BEDROCK_ACCESS_KEY") or None,
            bedrock_secret_key=os.getenv("AWS_BEDROCK_SECRET_KEY") or None,
            bedrock_max_tokens=int(os.getenv("METADATA_BEDROCK_MAX_TOKENS", "8192")),
            model_timeout_seconds=float(
                os.getenv("METADATA_MODEL_TIMEOUT_SECONDS", "45")
            ),
            request_timeout_seconds=float(
                os.getenv("METADATA_REQUEST_TIMEOUT_SECONDS", "120")
            ),
            max_request_bytes=int(os.getenv("METADATA_MAX_REQUEST_BYTES", "2097152")),
            max_concurrent_requests=int(
                os.getenv("METADATA_MAX_CONCURRENT_REQUESTS", "4")
            ),
            rate_limit_per_minute=int(
                os.getenv("METADATA_RATE_LIMIT_PER_MINUTE", "60")
            ),
            expose_docs=_bool("METADATA_EXPOSE_DOCS", False),
            model_provider=os.getenv(
                "METADATA_MODEL_PROVIDER",
                "bedrock",
            ).strip().lower(),
            jev_model=os.getenv(
                "METADATA_JEV_MODEL",
                "jev-latest",
            ).strip(),
            jev_api_key=os.getenv("TYPESAFE_API_KEY") or None,
            jev_transport=os.getenv(
                "METADATA_JEV_TRANSPORT",
                "typesafe",
            ).strip().lower(),

            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,

            openrouter_model=os.getenv(
                "METADATA_OPENROUTER_MODEL",
                "typesafe/jev-1.13",
            ).strip(),
        )

    def validate(self) -> None:
        if self.environment != "test" and not self.api_keys:
            raise ValueError(
                "METADATA_API_KEYS must contain at least one owner:secret entry"
            )

        for entry in self.api_keys:
            owner, separator, secret = entry.partition(":")
            if not separator or not owner.replace("-", "").replace("_", "").isalnum():
                raise ValueError("API keys must use owner-id:secret format")
            if self.environment != "test" and len(secret) < 32:
                raise ValueError(
                    "API key secrets must contain at least 32 characters"
                )

        if not self.allowed_hosts:
            raise ValueError("METADATA_ALLOWED_HOSTS cannot be empty")

        if self.max_concurrent_requests < 1 or self.rate_limit_per_minute < 1:
            raise ValueError("concurrency and rate limits must be positive")

        if self.model_provider not in {"bedrock", "jev"}:
            raise ValueError(
                "METADATA_MODEL_PROVIDER must be 'bedrock' or 'jev'"
            )

        if self.model_provider == "bedrock":
            if not self.bedrock_model and self.environment != "test":
                raise ValueError(
                    "METADATA_BEDROCK_MODEL_ID is required for Bedrock"
                )

            if bool(self.bedrock_access_key) != bool(self.bedrock_secret_key):
                raise ValueError(
                    "AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_KEY "
                    "must be provided together"
                )

        if self.model_provider == "jev":
            if self.jev_transport not in {"typesafe", "openrouter"}:
                raise ValueError(
                    "METADATA_JEV_TRANSPORT must be 'typesafe' or 'openrouter'"
                )

            if self.jev_transport == "typesafe":
                if not self.jev_api_key and self.environment != "test":
                    raise ValueError(
                        "TYPESAFE_API_KEY is required when using the "
                        "TypeSafe JEV transport"
                    )

            if self.jev_transport == "openrouter":
                if not self.openrouter_api_key and self.environment != "test":
                    raise ValueError(
                        "OPENROUTER_API_KEY is required when using the "
                        "OpenRouter JEV transport"
                    )

@lru_cache
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.validate()
    return settings
