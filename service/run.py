"""Console entrypoint to run the API server (``swargan-serve``)."""
import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "service.app:create_app",
        factory=True,
        host=os.environ.get("SWARGAN_HOST", "0.0.0.0"),
        port=int(os.environ.get("SWARGAN_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
