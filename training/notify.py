from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from dotenv import load_dotenv

load_dotenv()

TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "")

log = logging.getLogger(__name__)


def _post(payload: dict[str, Any]) -> None:
    if not TEAMS_WEBHOOK_URL:
        log.warning("TEAMS_WEBHOOK_URL not set; skipping notification")
        return
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        TEAMS_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status not in (200, 202):
            log.warning("Teams webhook returned HTTP %s", resp.status)


def notify_training_started(mode: str) -> None:
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "0078D4",
        "summary": "Training started",
        "sections": [
            {
                "activityTitle": "RL Prompt-Injection — Training Started",
                "facts": [
                    {"name": "Mode", "value": mode},
                ],
                "markdown": True,
            }
        ],
    }
    try:
        _post(payload)
    except Exception:
        log.warning("Failed to send Teams start notification", exc_info=True)


def notify_training_complete(mode: str, metrics: dict[str, Any]) -> None:
    ckpt_rates = metrics.get("ckpt_success_rates", [])
    final_rate = ckpt_rates[-1] if ckpt_rates else None
    rate_str = f"{final_rate:.1%}" if final_rate is not None else "n/a"

    ckpt_rewards = metrics.get("ckpt_mean_rewards", [])
    final_reward = ckpt_rewards[-1] if ckpt_rewards else None
    reward_str = f"{final_reward:.3f}" if final_reward is not None else "n/a"

    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "00B050",
        "summary": "Training completed",
        "sections": [
            {
                "activityTitle": "RL Prompt-Injection — Training Complete ✓",
                "facts": [
                    {"name": "Mode", "value": mode},
                    {"name": "Final success rate", "value": rate_str},
                    {"name": "Final mean reward", "value": reward_str},
                ],
                "markdown": True,
            }
        ],
    }
    try:
        _post(payload)
    except Exception:
        log.warning("Failed to send Teams completion notification", exc_info=True)


def notify_training_analysis(analysis_text: str) -> None:
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "0078D4",
        "summary": "Training analysis",
        "sections": [
            {
                "activityTitle": "RL Prompt-Injection — Post-Training Analysis",
                "text": analysis_text.replace("\n", "\n\n"),
                "markdown": False,
            }
        ],
    }
    try:
        _post(payload)
    except Exception:
        log.warning("Failed to send Teams analysis notification", exc_info=True)


def notify_training_failed(mode: str, error: BaseException) -> None:
    payload = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "FF0000",
        "summary": "Training failed",
        "sections": [
            {
                "activityTitle": "RL Prompt-Injection — Training Failed ✗",
                "facts": [
                    {"name": "Mode", "value": mode},
                    {"name": "Error", "value": f"{type(error).__name__}: {error}"},
                ],
                "markdown": True,
            }
        ],
    }
    try:
        _post(payload)
    except Exception:
        log.warning("Failed to send Teams failure notification", exc_info=True)
