import argparse
import json
import sys
import urllib.error
import urllib.request


def send_prompt(base_url: str, prompt: str, timeout: float) -> str:
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    if not isinstance(data, dict) or "response" not in data:
        raise ValueError(f"Unexpected response payload: {data}")

    return str(data["response"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive prompt loop client for target_bot /chat endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the target bot service",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    print(f"Connected client target: {args.base_url.rstrip('/')}/chat")
    print("Enter a prompt and press Enter. Ctrl+C to exit.")

    try:
        while True:
            try:
                prompt = input("\nYou> ").strip()
            except EOFError:
                print("\nEOF received. Exiting.")
                return 0

            if not prompt:
                continue

            try:
                response = send_prompt(args.base_url, prompt, args.timeout)
                print(f"Bot> {response}")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                print(f"Error: HTTP {exc.code} {detail}")
            except urllib.error.URLError as exc:
                print(f"Error: could not reach server ({exc})")
            except Exception as exc:
                print(f"Error: {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
