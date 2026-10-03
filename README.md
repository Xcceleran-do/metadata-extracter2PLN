# metadata-extractor2PLN

A headless, schema-adaptive metadata extraction service. It inspects JSON records, builds a typed extraction plan, applies deterministic and model-backed extractors, records evidence, and compiles accepted values into facts understood by PeTTaChainer.

The service does **not** execute MeTTa supplied by callers and does not write directly to a knowledge base. Its trust boundary ends at validated fact strings such as:

```metta
(: news_abc123 (engagement news_article-42 "High") (STV 0.75 0.9))
```

## What is implemented

* typed plan discovery with a deterministic fallback
* mandatory-property enforcement after model planning
* deterministic structured, numeric, date, length, reading-time, and engagement extraction
* Bedrock schema-constrained output for semantic properties
* JEV semantic classification for `semantic_text` properties
* JEV support through TypeSafe and OpenRouter transports
* allowed-value and source-evidence validation
* PeTTaChainer-compatible fact compilation with idempotency keys
* authenticated bulk HTTP endpoints with request, concurrency, timeout, and rate limits

The first release accepts bounded inline JSON batches. Durable jobs, source connectors, plan storage, and delivery to downstream PeTTaChainer servers are deliberately left outside this initial trust boundary.

Engagement is calculated from weighted interactions: comments count twice,
shares count three times, and other reactions count once. When views are
available the service classifies the resulting engagement rate; otherwise it
classifies the weighted interaction count. Compiled facts expose both the complete PeTTaChainer
statement and its validated `atom`/truth-value fields for downstream adapters.

Plan discovery sends only bounded samples to the configured model. If it is unavailable or
returns output that fails the service contract, discovery falls back to the
deterministic planner. Semantic properties are classified by the configured
semantic backend; each source text is capped at 12,000 characters before it
crosses the model boundary.

## JEV semantic classification

JEV is supported as a semantic-classification backend for properties using:

```text
extractor = "semantic_text"
```

For example:

```json
{
  "name": "sentiment",
  "description": "Classify the sentiment of the text.",
  "extractor": "semantic_text",
  "field_paths": [],
  "allowed_values": [
    "positive",
    "negative",
    "neutral"
  ],
  "required": true,
  "include_in_pln": true
}
```

The JEV backend uses the `allowed_values` from the extraction plan as the
classification choices. The returned classification is validated against the
same extraction plan before it is accepted and compiled into a
PeTTaChainer-compatible fact.

JEV is currently used for **semantic classification only**. It does not
perform generic structured generation or extraction-plan discovery.

When JEV is configured, plan discovery uses the deterministic planner
fallback.

### JEV with OpenRouter

JEV can be accessed through OpenRouter using the JEV model:

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=openrouter
OPENROUTER_API_KEY=replace-with-your-openrouter-key
METADATA_OPENROUTER_MODEL=typesafe/jev-1.13
```

The OpenRouter transport sends classification requests directly to the
OpenRouter JEV decision endpoint.

### JEV with TypeSafe

JEV can also be accessed through the TypeSafe SDK:

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=typesafe
TYPESAFE_API_KEY=replace-with-your-typesafe-key
METADATA_JEV_MODEL=jev-latest
```

The TypeSafe transport uses the `typesafe-sdk` package.

### JEV classification behavior

For each semantic property, the backend provides JEV with:

* the source text
* the property description
* the allowed classification values
* the configured JEV model

JEV returns the selected classification together with its probability/confidence
information. The extraction service then validates the result before producing
the final metadata and PeTTaChainer fact.

For example, a successful classification can produce a fact such as:

```metta
(: test_25d7ff74fb9a97da
   (sentiment test_Excellent_article "positive")
   (STV 1 1))
```

Provider credentials must be supplied through environment variables and must
not be committed to the repository.

## Run locally

Python 3.11 or newer and `uv` are recommended.

```bash
cp .env.example .env
# Edit .env, set a long API secret, choose a model provider, and set its key.
uv sync --extra dev
set -a; source .env; set +a
uv run uvicorn metadata_extractor2pln.api:app --host 127.0.0.1 --port 8080
```

`METADATA_API_KEYS` is a comma-separated list so multiple clients can be rotated independently. Each entry is `owner-id:secret`; callers send only the secret as the bearer token. It is unrelated to model-provider authentication.

Bedrock is the default provider. Choose a model that supports Bedrock structured
outputs, such as DeepSeek V3.2:

```dotenv
METADATA_BEDROCK_MODEL_ID=deepseek.v3.2
AWS_REGION=us-east-1
```

### Selecting the model provider

The service supports both Bedrock and JEV:

```dotenv
METADATA_MODEL_PROVIDER=bedrock
```

or:

```dotenv
METADATA_MODEL_PROVIDER=jev
```

Bedrock remains the default provider.

### Using JEV through OpenRouter

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=openrouter
OPENROUTER_API_KEY=replace-with-your-openrouter-key
METADATA_OPENROUTER_MODEL=typesafe/jev-1.13
```

### Using JEV through TypeSafe

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=typesafe
TYPESAFE_API_KEY=replace-with-your-typesafe-key
METADATA_JEV_MODEL=jev-latest
```

```bash
curl -s http://127.0.0.1:8080/health

curl -s http://127.0.0.1:8080/v1/run \
  -H 'Authorization: Bearer replace-with-at-least-32-random-characters' \
  -H 'Content-Type: application/json' \
  -d '{
    "namespace": "demo",
    "source_name": "articles",
    "records": [{
      "id": "article-42",
      "content": "A concise technical introduction for experienced engineers.",
      "likes": 18,
      "comments": 4
    }]
  }'
```

For a completely offline smoke test, set `"use_model": false` and `"required_properties": ["engagement"]`. Omit `required_properties` to let plan discovery choose properties from the supplied records; provide it only when specific properties must be present.

## API

* `GET /health` — process liveness, no authentication
* `GET /ready` — service and model-backend readiness, no authentication
* `POST /v1/plans/discover` — inspect samples and return a sanitized plan
* `POST /v1/plans/validate` — verify a pinned plan and its fingerprint
* `POST /v1/extract` — apply a previously generated plan to up to 100 records
* `POST /v1/run` — plan and extract one bounded batch

Generated plans contain a fingerprint. Changing a plan without regenerating its fingerprint causes `/v1/extract` to reject it. This prevents an audited plan from silently changing in transit.

## Model planning and extraction

The service separates extraction-plan discovery from semantic classification.

### Plan discovery

Plan discovery sends only bounded samples to the configured model. If the
model is unavailable or returns output that fails the service contract,
discovery falls back to the deterministic planner.

JEV does not perform plan discovery.

### Property extraction

After a plan has been validated, properties can be extracted using the
configured deterministic or model-backed extractor.

Supported extractor types include:

* `structured_field`
* `calculated_metric`
* `numeric_bucket`
* `date_bucket`
* `semantic_text`

`semantic_text` properties can be handled by the configured Bedrock or JEV
semantic backend.

Each source text is capped at 12,000 characters before it crosses the model
boundary.

## Semantic evidence validation

Semantic model outputs are validated after the model response is received.

Allowed values are constrained by the extraction plan. A classification that
does not correspond to an allowed value is rejected rather than silently
accepted.

When the model provides an evidence quote, the quote is checked against the
original source text. If the returned quote cannot be verified, it is removed
and the associated confidence is bounded accordingly.

The service therefore does not treat model output as trusted simply because
the model returned it.

## PeTTaChainer facts

Accepted extracted values are compiled into PeTTaChainer-compatible facts.

For example:

```metta
(: test_25d7ff74fb9a97da
   (sentiment test_Excellent_article "positive")
   (STV 1 1))
```

Compiled facts expose:

* the complete PeTTaChainer statement
* the extracted atom
* strength
* confidence
* proof ID
* idempotency key
* property name

This allows downstream adapters to consume the extracted knowledge without
having to trust or execute arbitrary model output.

## Development

```bash
uv run pytest
```

The optional integration test validates compiled facts with the sibling PeTTaChainer checkout when that package is available.

The JEV backend is covered by tests for:

* TypeSafe JEV classification
* missing API-key handling
* plan-discovery rejection
* required `allowed_values`
* OpenRouter transport behavior
* returned classification, probability, confidence, and usage values

## Current security boundary

* JSON only; no arbitrary URL fetching or user-supplied code
* strict Pydantic request and model-output schemas
* post-model enforcement of required properties and known source paths
* allowed-value validation for semantic classifications
* source-evidence quote verification
* bearer authentication
* per-owner rate limiting
* bounded request body size
* bounded record count
* bounded concurrency
* model request timeouts
* no execution of caller-supplied MeTTa
* no direct knowledge-base mutation
* no persistence or downstream mutation in v0.1
* model-provider credentials are supplied through environment variables

The service remains behind the PeTTaChainer boundary. It produces validated
fact strings rather than directly mutating the knowledge base.

## Dependencies

The JEV update adds the following runtime dependencies:

```text
typesafe-sdk>=0.7.2
httpx>=0.28.1
```

`typesafe-sdk` is used for the direct TypeSafe JEV transport.

`httpx` is used for the OpenRouter JEV transport because the OpenRouter JEV
decision API is accessed directly rather than through the TypeSafe SDK.

Development dependencies include `pytest`.

## Configuration reference

### General

```dotenv
METADATA_MODEL_PROVIDER=bedrock
METADATA_API_KEYS=owner-id:replace-with-a-long-random-secret
```

### Bedrock

```dotenv
METADATA_BEDROCK_MODEL_ID=deepseek.v3.2
AWS_REGION=us-east-1
```

### JEV / TypeSafe

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=typesafe
METADATA_JEV_MODEL=jev-latest
TYPESAFE_API_KEY=replace-with-your-typesafe-key
```

### JEV / OpenRouter

```dotenv
METADATA_MODEL_PROVIDER=jev
METADATA_JEV_TRANSPORT=openrouter
OPENROUTER_API_KEY=replace-with-your-openrouter-key
METADATA_OPENROUTER_MODEL=typesafe/jev-1.13
```

Never place real provider API keys in source code, README examples, tests, or
committed `.env` files.

## Operational behavior

The service is intentionally bounded around inline JSON batches.

Request processing is protected by:

* request size limits
* record-count limits
* concurrency limits
* timeout limits
* rate limits
* strict request validation
* strict model-output validation

These limits prevent model-backed extraction from becoming an unbounded
execution path.

The service can report partial failures for individual records while
preserving successfully validated results.

## Deployment considerations

For deployment, configure:

* a strong `METADATA_API_KEYS` secret
* the appropriate model-provider credentials
* model request timeouts
* request and concurrency limits
* rate limits
* TLS at the deployment boundary
* production logging and monitoring
* health and readiness checks

Bedrock and JEV credentials should be managed as deployment secrets rather
than stored in the repository.

The service is designed to remain a bounded metadata-conversion component.
Durable jobs, source connectors, persistent plan storage, and direct delivery
to downstream PeTTaChainer servers are outside the initial v0.1 trust boundary.
