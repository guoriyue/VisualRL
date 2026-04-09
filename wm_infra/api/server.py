"""Compatibility entrypoint for the Gateway app."""

from wm_infra.gateway.app import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
