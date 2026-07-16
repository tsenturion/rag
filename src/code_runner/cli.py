from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "code_runner.app:app",
        host=os.getenv("CODE_RUNNER_HOST", "127.0.0.1"),
        port=int(os.getenv("CODE_RUNNER_PORT", "8010")),
        workers=1,
    )


if __name__ == "__main__":
    main()
