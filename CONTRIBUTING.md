# Contributing

## Development Setup

```powershell
python -m pip install -e .
python -m unittest discover -v
```

The project intentionally uses only the Python standard library at runtime.

## Fixture Hygiene

LM Studio logs can include local paths, usernames, prompts, request bodies, tool calls, and assistant responses. Before adding or sharing fixtures:

- Prefer compact synthetic or minimized logs under `tests/fixtures/`.
- Remove local user-profile paths and machine-specific directories.
- Remove request bodies, chat messages, tool-call payloads, and assistant reasoning/output.
- Keep fixtures just large enough to exercise the parser behavior under test.

## Pull Requests

- Add or update tests for parser behavior changes.
- Run `python -m unittest discover -v` before submitting.
- Keep unrelated formatting and refactors out of focused fixes.
