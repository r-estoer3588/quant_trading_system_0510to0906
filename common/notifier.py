"""通知ユーティリティ。

- Slack / Discord Webhook に対応
- 日本語の文言と絵文字を正しく整形して送信

注意:
- このモジュールは UTF-8 で保存されています。
- Windows コンソール（cp932）の制限で絵文字が表示できない場合がありますが、
  Slack/Discord やログファイル（UTF-8）では問題ありません。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

try:  # pragma: no cover - optional dependency
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore
except Exception:  # pragma: no cover - missing optional dependency
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore

__all__ = [
    "Notifier",
    "BroadcastNotifier",
    "FallbackNotifier",
    "create_notifier",
    "now_jst_str",
    "mask_secret",
    "truncate",
    "format_table",
    "chunk_fields",
    "detect_default_platform",
    "get_notifiers_from_env",
]


SYSTEM_POSITION: dict[str, str] = {
    "system1": "long",
    "system2": "short",
    "system3": "long",
    "system4": "long",
    "system5": "long",
    "system6": "short",
    "system7": "short",
}

COLOR_LONG = 0x2ECC71
COLOR_SHORT = 0xE74C3C
COLOR_NEUTRAL = 0xF1C40F

_JST = ZoneInfo("Asia/Tokyo")


class _JSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):  # type: ignore[override]
        dt = datetime.fromtimestamp(record.created, tz=_JST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M JST")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("notifier")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = "[%(asctime)s] %(levelname)s Notifier: %(message)s"
    formatter = _JSTFormatter(fmt)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    now = datetime.now(tz=_JST)
    log_file = logs_dir / f"notifier_{now:%Y-%m}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def now_jst_str(minute: bool = True) -> str:
    fmt = "%Y-%m-%d %H:%M JST" if minute else "%Y-%m-%d %H:%M:%S JST"
    return datetime.now(tz=_JST).strftime(fmt)


def mask_secret(url: str) -> str:
    if not url:
        return ""
    try:
        head, tail = url.split("://", 1)
        domain, *rest = tail.split("/")
        token = "/".join(rest)
        if len(token) > 9:
            token = f"{token[:5]}...{token[-4:]}"
        else:
            token = "***"
        return f"{head}://{domain}/{token}"
    except Exception:
        return "***"


def truncate(text: Any, max_len: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= max_len else s[:max_len] + "… (truncated)"


def format_table(
    rows: list[Iterable[Any]], headers: list[str] | None = None, max_width: int = 80
) -> str:
    if not rows:
        return ""
    data = [list(map(str, r)) for r in rows]
    if headers:
        data.insert(0, list(map(str, headers)))
    cols = len(data[0])
    widths = [max(len(r[i]) for r in data) for i in range(cols)]
    total = sum(widths) + 3 * (cols - 1)
    if total > max_width:
        ratio = (max_width - 3 * (cols - 1)) / sum(widths)
        widths = [max(1, int(w * ratio)) for w in widths]

    def fmt_row(r: list[str]) -> str:
        return " | ".join(s[: widths[i]].ljust(widths[i]) for i, s in enumerate(r))

    lines: list[str] = []
    if headers:
        lines.append(fmt_row(data[0]))
        lines.append("-+-".join("-" * w for w in widths))
        body = data[1:]
    else:
        body = data
    for r in body:
        lines.append(fmt_row(r))
    return "```\n" + "\n".join(lines) + "\n```"


def chunk_fields(
    name: str, items: list[str], inline: bool = True, max_per_field: int = 15
) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    if not items:
        return fields
    for i in range(0, len(items), max_per_field):
        chunk = [str(x) for x in items[i : i + max_per_field]]
        fields.append(
            {
                "name": name if i == 0 else f"{name} ({i // max_per_field + 1})",
                "value": "\n".join(chunk),
                "inline": inline,
            }
        )
    return fields


def detect_default_platform() -> str:
    if os.getenv("SLACK_WEBHOOK_URL") or os.getenv("SLACK_BOT_TOKEN"):
        return "slack"
    if os.getenv("DISCORD_WEBHOOK_URL"):
        return "discord"
    return "none"


def _notifications_disabled() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    flag = (os.getenv("CI") or "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    flag2 = (os.getenv("DISABLE_NOTIFICATIONS") or "").strip().lower()
    return flag2 in {"1", "true", "yes"}


class Notifier:
    def __init__(self, platform: str = "auto", webhook_url: str | None = None):
        if platform == "auto":
            platform = detect_default_platform()
        self.platform = platform
        self.webhook_url = webhook_url or (
            os.getenv("SLACK_WEBHOOK_URL")
            if platform == "slack"
            else os.getenv("DISCORD_WEBHOOK_URL")
        )
        self.logger = _setup_logger()

    def _post(self, payload: dict[str, Any]) -> None:
        if _notifications_disabled():
            self.logger.info("通知送信は無効化されています（テスト/CI/環境変数）")
            return
        if getattr(self, "platform", "") == "slack":
            if os.getenv("SLACK_BOT_TOKEN", "").strip():
                ok = False
                try:
                    ok = self._post_slack_api(payload)
                except Exception as e:  # pragma: no cover
                    self.logger.warning("Slack API exception: %s", e)
                if not ok:
                    self.logger.error(
                        "Slack API送信に失敗しました（Webhookへはフォールバックしません）"
                    )
                return
        if not self.webhook_url:
            self.logger.warning(
                "webhook 未設定のため送信をスキップします platform=%s", self.platform
            )
            return
        url = self.webhook_url
        masked = mask_secret(url)
        try:
            r = requests.post(url, json=payload, timeout=10)
            if 200 <= r.status_code < 300:
                return
            self.logger.warning("送信失敗 status=%s body=%s", r.status_code, truncate(r.text, 100))
        except Exception as e:  # pragma: no cover
            self.logger.warning("送信エラー %s", e)
        self.logger.error("送信に失敗しました: %s", masked)
        raise RuntimeError("notification failed")

    def _post_slack_api(self, payload: dict[str, Any]) -> bool:
        token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        channel = (
            payload.pop("_channel", None)
            or os.getenv("SLACK_CHANNEL", "").strip()
            or os.getenv("SLACK_CHANNEL_ID", "").strip()
        )
        if not token or not channel or WebClient is None:
            return False
        blocks = payload.get("blocks")
        text = payload.get("text") or "Notification"
        try:  # pragma: no cover
            client = WebClient(token=token)  # type: ignore
            client.chat_postMessage(channel=channel, text=text, blocks=blocks)  # type: ignore
            self.logger.info("sent via Slack Web API to channel=%s", channel)
            return True
        except SlackApiError as e:  # type: ignore[name-defined]
            try:
                msg = e.response.get("error") if hasattr(e, "response") else str(e)
            except Exception:
                msg = str(e)
            self.logger.warning("Slack API error: %s", truncate(msg, 200))
            return False
        except Exception as e:  # pragma: no cover
            self.logger.warning("Slack API exception: %s", e)
            return False

    # 共通 send の簡易版
    def send(
        self,
        title: str,
        message: str,
        fields: dict[str, str] | list[dict[str, Any]] | None = None,
        image_url: str | None = None,
        color: int | None = None,
        channel: str | None = None,
    ) -> None:
        desc = f"実行時刻 {now_jst_str()}"
        if message:
            desc += "\n" + message
        payload: dict[str, Any]
        if self.platform == "discord":
            embed: dict[str, Any] = {
                "title": truncate(title, 256),
                "description": truncate(desc, 4096),
            }
            if color is not None:
                embed["color"] = int(color)
            field_list: list[dict[str, Any]] = []
            if isinstance(fields, dict):
                for k, v in fields.items():
                    field_list.append(
                        {"name": truncate(k, 256), "value": truncate(str(v), 1024), "inline": True}
                    )
            elif isinstance(fields, list):
                for f in fields:
                    field_list.append(
                        {
                            "name": truncate(f.get("name", ""), 256),
                            "value": truncate(str(f.get("value", "")), 1024),
                            "inline": bool(f.get("inline", True)),
                        }
                    )
            if field_list:
                embed["fields"] = field_list[:25]
            if image_url:
                embed["image"] = {"url": image_url}
            payload = {"embeds": [embed]}
        else:  # slack/none
            blocks: list[dict[str, Any]] = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": truncate(f"*{title}*\n{desc}", 3000)},
                }
            ]
            if isinstance(fields, dict):
                text = "\n".join(f"*{k}*: {v}" for k, v in fields.items())
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": truncate(text, 3000)}}
                )
            elif isinstance(fields, list):
                for f in fields:
                    text = f"*{f.get('name', '')}*\n{f.get('value', '')}"
                    blocks.append(
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": truncate(text, 3000)},
                        }
                    )
            if image_url:
                blocks.append({"type": "image", "image_url": image_url, "alt_text": title})
            fallback = truncate(f"{title}\n{desc}", 3000)
            payload = {"text": fallback, "blocks": blocks}
        self.logger.info(
            "send title=%s fields=%d image=%s",
            truncate(title, 50),
            0 if not fields else (len(fields) if isinstance(fields, list) else len(fields)),
            bool(image_url),
        )
        if channel:
            payload["_channel"] = channel
        self._post(payload)

    # メンション対応
    def send_with_mention(
        self,
        title: str,
        message: str,
        fields: dict[str, str] | list[dict[str, Any]] | None = None,
        image_url: str | None = None,
        color: int | None = None,
        mention: str | bool | None = None,
        channel: str | None = None,
    ) -> None:
        desc = f"実行時刻 {now_jst_str()}"
        if message:
            desc += "\n" + message
        content: str | None = None
        if mention is None:
            _m = os.getenv("NOTIFY_MENTION", "").strip().lower()
            if _m in {"channel", "here", "@everyone", "@here"}:
                mention = _m
        if mention:
            if self.platform == "slack":
                tag = (
                    "<!channel>" if str(mention).lower() in {"channel", "@everyone"} else "<!here>"
                )
                desc = f"{tag}\n" + desc
            else:
                content = (
                    "@everyone" if str(mention).lower() in {"channel", "@everyone"} else "@here"
                )

        payload: dict[str, Any]
        if self.platform == "discord":
            embed: dict[str, Any] = {
                "title": truncate(title, 256),
                "description": truncate(desc, 4096),
            }
            if color is not None:
                embed["color"] = int(color)
            field_list: list[dict[str, Any]] = []
            if isinstance(fields, dict):
                for k, v in fields.items():
                    field_list.append(
                        {"name": truncate(k, 256), "value": truncate(str(v), 1024), "inline": True}
                    )
            elif isinstance(fields, list):
                for f in fields:
                    field_list.append(
                        {
                            "name": truncate(f.get("name", ""), 256),
                            "value": truncate(str(f.get("value", "")), 1024),
                            "inline": bool(f.get("inline", True)),
                        }
                    )
            if field_list:
                embed["fields"] = field_list[:25]
            if image_url:
                embed["image"] = {"url": image_url}
            payload = {"embeds": [embed]}
            if content:
                payload["content"] = content
        else:
            blocks: list[dict[str, Any]] = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": truncate(f"*{title}*\n{desc}", 3000)},
                }
            ]
            if isinstance(fields, dict):
                text = "\n".join(f"*{k}*: {v}" for k, v in fields.items())
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": truncate(text, 3000)}}
                )
            elif isinstance(fields, list):
                for f in fields:
                    text = f"*{f.get('name', '')}*\n{f.get('value', '')}"
                    blocks.append(
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": truncate(text, 3000)},
                        }
                    )
            if image_url:
                blocks.append({"type": "image", "image_url": image_url, "alt_text": title})
            fallback = truncate(f"{title}\n{desc}", 3000)
            payload = {"text": fallback, "blocks": blocks}
        self.logger.info(
            "send+mention title=%s fields=%d image=%s",
            truncate(title, 50),
            0 if not fields else (len(fields) if isinstance(fields, list) else len(fields)),
            bool(image_url),
        )
        if channel:
            payload["_channel"] = channel
        self._post(payload)

    def send_signals(
        self, system_name: str, signals: list[str], *, channel: str | None = None
    ) -> None:
        direction = SYSTEM_POSITION.get(system_name.lower(), "")
        color = (
            COLOR_LONG
            if direction == "long"
            else COLOR_SHORT if direction == "short" else COLOR_NEUTRAL
        )
        title = f"📢 {system_name} 日次シグナル ・ {now_jst_str()}"
        ch = channel or (os.getenv("SLACK_CHANNEL_SIGNALS") if self.platform == "slack" else None)
        if not signals:
            self.send(title, "本日のシグナルはありません", color=color, channel=ch)
            self.logger.info("signals %s direction=%s count=0", system_name, direction or "none")
            return
        emoji = "🟢" if direction == "long" else ("🔴" if direction == "short" else "")
        items = [f"{emoji} {s}" if emoji else s for s in signals]
        fields = chunk_fields("銘柄", items, inline=False)
        self.send(title, "", fields=fields, color=color, channel=ch)
        self.logger.info(
            "signals %s direction=%s count=%d", system_name, direction or "none", len(signals)
        )

    def send_backtest(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[str],
        *,
        channel: str | None = None,
    ) -> None:
        period_with_run = (
            f"{period}, 実行日 ・ {now_jst_str()}" if period else f"実行日 ・ {now_jst_str()}"
        )
        self.send_backtest_ex(system_name, period_with_run, stats, ranking, channel=channel)
        summary = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:3])
        self.logger.info("backtest %s stats=%s top=%d", system_name, summary, min(len(ranking), 10))

    def send_trade_report(self, system_name: str, trades: list[dict[str, Any]]) -> None:
        title = f"🧾 {system_name} 売買結果 ・ {now_jst_str()}"
        if not trades:
            self.send(title, "本日の売買はありません")
            self.logger.info("trade report %s count=0", system_name)
            return
        rows = []
        total = 0.0
        for t in trades:
            sym = str(t.get("symbol"))
            action = str(t.get("action", t.get("side", ""))).upper()
            qty = t.get("qty", t.get("shares", 0))
            price = float(t.get("price", t.get("entry_price", 0.0) or 0.0))
            try:
                total += float(qty) * float(price)
            except Exception:
                pass
            rows.append([sym, action, f"{qty}", f"@{price:.4f}"])
        table = format_table(rows, headers=["SYMBOL", "ACTION", "QTY", "PRICE"])
        self.send(title, table)
        self.logger.info("trade report %s count=%d notional=%.2f", system_name, len(trades), total)

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
    ) -> None:
        title = (
            f"📊 {system_name} {period_type} サマリー ・ {period_label}, 実行日 ・ {now_jst_str()}"
        )
        fields = {k: str(v) for k, v in summary.items()}
        self.send(title, "", fields=fields, image_url=image_url)
        self.logger.info("summary %s %s keys=%d", system_name, period_type, len(summary))

    def send_backtest_ex(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[Any],
        image_url: str | None = None,
        mention: str | bool | None = None,
        *,
        channel: str | None = None,
    ) -> None:
        direction = SYSTEM_POSITION.get(system_name.lower(), "")
        color = (
            COLOR_LONG
            if direction == "long"
            else COLOR_SHORT if direction == "short" else COLOR_NEUTRAL
        )
        title = f"📊 {system_name} バックテスト ・ {period}"
        fields = {k: str(v) for k, v in stats.items()}
        desc = ""
        if ranking:
            lines: list[str] = []
            for i, item in enumerate(ranking[:10], start=1):
                try:
                    if isinstance(item, dict):
                        sym = item.get("symbol") or item.get("sym") or item.get("ticker") or "?"
                        roc = item.get("roc")
                        vol = item.get("volume") or item.get("vol")
                        part = f"{sym}"
                        if roc is not None:
                            part += f"  ROC200:{float(roc):.2f}"
                        if vol is not None:
                            part += f"  Vol:{int(float(vol)):,}"
                        lines.append(f"{i}. {part}")
                    else:
                        lines.append(f"{i}. {item}")
                except Exception:
                    lines.append(f"{i}. {item}")
            if len(ranking) > 10:
                lines.append("…")
            desc = "ROC200 TOP10\n" + "\n".join(lines)
        if mention and getattr(self, "platform", "") == "slack":
            tag = "<!channel>" if str(mention).lower() in {"channel", "@everyone"} else "<!here>"
            desc = f"{tag}\n" + desc
        ch = channel or (os.getenv("SLACK_CHANNEL_EQUITY") if self.platform == "slack" else None)
        self.send(title, desc, fields=fields, color=color, image_url=image_url, channel=ch)
        summary = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:3])
        self.logger.info(
            "backtest_ex %s stats=%s top=%d", system_name, summary, min(len(ranking), 10)
        )


class BroadcastNotifier:
    def __init__(self, notifiers: list[Notifier]) -> None:
        self._notifiers = [n for n in notifiers if getattr(n, "webhook_url", None)]
        self.logger = _setup_logger()

    def _each(self, fn_name: str, *args, **kwargs) -> None:
        """
        各 Notifier を登録順に試し、最初に成功した Notifier で処理を終了する。
        失敗はログに記録し、すべて失敗した場合は警告ログを出すのみ（例外は伝播しない）。
        """
        any_succeeded = False
        for n in self._notifiers:
            platform = getattr(n, "platform", "?")
            try:
                getattr(n, fn_name)(*args, **kwargs)
                self.logger.info("broadcast %s succeeded platform=%s", fn_name, platform)
                any_succeeded = True
                break  # 成功したら以降の通知は行わない（Slack成功時はDiscordに送らない）
            except Exception as e:  # pragma: no cover
                self.logger.warning("broadcast %s failed platform=%s %s", fn_name, platform, e)
                # 継続して次の Notifier（例: Slack失敗時にDiscordへ）を試す

        if not any_succeeded:
            self.logger.warning("broadcast %s: all notifiers failed", fn_name)

    def send(self, *args, **kwargs) -> None:
        self._each("send", *args, **kwargs)

    def send_signals(self, *args, **kwargs) -> None:
        self._each("send_signals", *args, **kwargs)

    def send_backtest(self, *args, **kwargs) -> None:
        self._each("send_backtest", *args, **kwargs)

    def send_backtest_ex(self, *args, **kwargs) -> None:
        self._each("send_backtest_ex", *args, **kwargs)

    def send_trade_report(self, *args, **kwargs) -> None:
        self._each("send_trade_report", *args, **kwargs)

    def send_summary(self, *args, **kwargs) -> None:
        self._each("send_summary", *args, **kwargs)


class FallbackNotifier:
    def __init__(self) -> None:
        self._logger = _setup_logger()
        self._slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        self._slack_default_ch = (
            os.getenv("SLACK_CHANNEL", "").strip() or os.getenv("SLACK_CHANNEL_ID", "").strip()
        )
        try:
            discord_url = os.getenv("DISCORD_WEBHOOK_URL")
            self._discord = (
                Notifier(platform="discord", webhook_url=discord_url) if discord_url else None
            )
        except Exception:
            self._discord = None

    def _slack_send_text(self, text: str, *, channel: str | None = None) -> bool:
        if _notifications_disabled():
            self._logger.info("通知送信は無効化されています（テスト/CI/環境変数）")
            return True
        ch = channel or self._slack_default_ch
        if not ch:
            return False
        token = self._slack_token
        webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
        if token and WebClient is not None:
            try:  # pragma: no cover
                client = WebClient(token=token)  # type: ignore
                client.chat_postMessage(channel=ch, text=text)  # type: ignore
                self._logger.info("fallback: sent via Slack API to %s", ch)
                return True
            except SlackApiError as e:  # type: ignore[name-defined]
                try:
                    msg = e.response.get("error") if hasattr(e, "response") else str(e)
                except Exception:
                    msg = str(e)
                self._logger.warning("fallback: Slack API error: %s", truncate(msg, 200))
            except Exception as e:
                self._logger.warning("fallback: Slack API exception: %s", e)
        if webhook:
            try:  # pragma: no cover
                r = requests.post(webhook, json={"text": text}, timeout=10)
                if 200 <= r.status_code < 300:
                    self._logger.info("fallback: sent via Slack Webhook")
                    return True
                self._logger.warning(
                    "fallback: webhook status=%s body=%s", r.status_code, truncate(r.text, 200)
                )
            except Exception as e:
                self._logger.warning("fallback: Slack webhook exception: %s", e)
        return False

    def _slack_upload_file(
        self, image_path: str, *, title: str, initial_comment: str, channel: str | None
    ) -> bool:
        if _notifications_disabled():
            return True
        token = self._slack_token
        ch = channel or self._slack_default_ch
        if not token or not ch or WebClient is None:
            return False
        try:  # pragma: no cover
            client = WebClient(token=token)  # type: ignore
            client.files_upload_v2(  # type: ignore
                channel=ch,
                initial_comment=initial_comment,
                title=title,
                file=image_path,
            )
            self._logger.info("fallback: file uploaded via Slack API to %s", ch)
            return True
        except SlackApiError as e:  # type: ignore[name-defined]
            try:
                msg = e.response.get("error") if hasattr(e, "response") else str(e)
            except Exception:
                msg = str(e)
            self._logger.warning("fallback: Slack file upload error: %s", truncate(msg, 200))
            return False
        except Exception as e:
            self._logger.warning("fallback: Slack file upload exception: %s", e)
            return False

    def _discord_call(self, fn_name: str, *args, **kwargs) -> bool:
        if not self._discord:
            return False
        try:
            getattr(self._discord, fn_name)(*args, **kwargs)
            self._logger.info("fallback: sent via Discord (%s)", fn_name)
            return True
        except Exception as e:  # pragma: no cover
            self._logger.warning("fallback: Discord send failed (%s) %s", fn_name, e)
            return False

    def send(
        self,
        title: str,
        message: str,
        fields: dict[str, str] | list[dict[str, Any]] | None = None,
        image_url: str | None = None,
        color: int | None = None,
        channel: str | None = None,
    ) -> None:  # noqa: E501
        lines = [f"{title}"]
        if message:
            lines.append(str(message))
        if isinstance(fields, dict) and fields:
            kv = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:10])
            lines.append(kv)
        text = "\n".join(lines)
        ch = channel or os.getenv("SLACK_CHANNEL_LOGS") or None
        if self._slack_send_text(text, channel=ch):
            return
        if not self._discord_call(
            "send", title, message, fields=fields, image_url=image_url, color=color
        ):
            raise RuntimeError("notification failed (slack+discord)")

    def send_with_mention(
        self,
        title: str,
        message: str,
        fields: dict[str, str] | list[dict[str, Any]] | None = None,
        image_url: str | None = None,
        color: int | None = None,
        mention: str | bool | None = None,
        channel: str | None = None,
        image_path: str | None = None,
    ) -> None:  # noqa: E501
        tag = None
        if mention:
            tag = "@everyone" if str(mention).lower() in {"channel", "@everyone"} else "@here"
        text = (
            f"{('@' + tag.split('@')[-1]) + ' ' if tag else ''}{title}\n{message}"
            if message
            else f"{('@' + tag.split('@')[-1]) + ' ' if tag else ''}{title}"
        )
        ch = channel or os.getenv("SLACK_CHANNEL_EQUITY") or None
        if image_path and self._slack_upload_file(
            image_path, title=title, initial_comment=text, channel=ch
        ):
            return
        if self._slack_send_text(text, channel=ch):
            return
        if not self._discord_call(
            "send_with_mention",
            title,
            message,
            fields=fields,
            image_url=image_url,
            color=color,
            mention=mention,
        ):  # noqa: E501
            raise RuntimeError("notification failed (slack+discord)")

    def send_signals(
        self, system_name: str, signals: list[str], *, channel: str | None = None
    ) -> None:
        title = f"📢 {system_name} 日次シグナル ・ {now_jst_str()}"
        if not signals:
            text = f"{title}\n本日のシグナルはありません"
        else:
            top = [str(s) for s in signals[:50]]
            text = f"{title}\n" + "\n".join(top)
        ch = channel or os.getenv("SLACK_CHANNEL_SIGNALS") or None
        if self._slack_send_text(text, channel=ch):
            return
        if not self._discord_call("send_signals", system_name, signals):
            raise RuntimeError("notification failed (slack+discord)")

    def send_backtest(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[str],
        *,
        channel: str | None = None,
    ) -> None:  # noqa: E501
        period_with_run = (
            f"{period}, 実行日 ・ {now_jst_str()}" if period else f"実行日 ・ {now_jst_str()}"
        )
        self.send_backtest_ex(system_name, period_with_run, stats, ranking, channel=channel)

    def send_backtest_ex(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[Any],
        image_url: str | None = None,
        mention: str | bool | None = None,
        *,
        channel: str | None = None,
        image_path: str | None = None,
    ) -> None:  # noqa: E501
        title = f"📊 {system_name} バックテスト ・ {period}"
        summary = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:5])
        lines = [title]
        if summary:
            lines.append(summary)
        if ranking:
            top = []
            for i, item in enumerate(ranking[:10], start=1):
                try:
                    if isinstance(item, dict):
                        sym = item.get("symbol") or item.get("sym") or item.get("ticker") or "?"
                        roc = item.get("roc")
                        vol = item.get("volume") or item.get("vol")
                        part = f"{sym}"
                        if roc is not None:
                            part += f" ROC200:{float(roc):.2f}"
                        if vol is not None:
                            part += f" Vol:{int(float(vol)):,}"
                        top.append(f"{i}. {part}")
                    else:
                        top.append(f"{i}. {item}")
                except Exception:
                    top.append(f"{i}. {item}")
            lines.append("\n".join(top))
        ch = channel or os.getenv("SLACK_CHANNEL_EQUITY") or None
        text = "\n".join(lines)
        if image_path and self._slack_upload_file(
            image_path, title=title, initial_comment=text, channel=ch
        ):
            return
        if self._slack_send_text(text, channel=ch):
            return
        if not self._discord_call(
            "send_backtest_ex",
            system_name,
            period,
            stats,
            ranking,
            image_url=image_url,
            mention=mention,
        ):  # noqa: E501
            raise RuntimeError("notification failed (slack+discord)")

    def send_trade_report(self, system_name: str, trades: list[dict[str, Any]]) -> None:
        title = f"🧾 {system_name} 売買結果 ・ {now_jst_str()}"
        if not trades:
            text = f"{title}\n本日の売買はありません"
        else:
            rows = []
            for t in trades:
                sym = str(t.get("symbol"))
                side = str(t.get("action", t.get("side", "")).upper())
                qty = t.get("qty", t.get("shares", 0))
                price = t.get("price", t.get("entry_price", 0.0) or 0.0)
                rows.append(f"{sym} {side} {qty} @{float(price):.4f}")
            text = f"{title}\n" + "\n".join(rows[:20])
        if self._slack_send_text(text):
            return
        if not self._discord_call("send_trade_report", system_name, trades):
            raise RuntimeError("notification failed (slack+discord)")

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
    ) -> None:  # noqa: E501
        title = (
            f"📊 {system_name} {period_type} サマリー ・ {period_label}, 実行日 ・ {now_jst_str()}"
        )
        kv = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:10])
        text = f"{title}\n{kv}" if kv else title
        if self._slack_send_text(text):
            return
        if not self._discord_call(
            "send_summary", system_name, period_type, period_label, summary, image_url=image_url
        ):  # noqa: E501
            raise RuntimeError("notification failed (slack+discord)")


def create_notifier(
    platform: str = "auto", broadcast: bool | None = None, fallback: bool | None = None
):
    if broadcast is None:
        flag = os.getenv("NOTIFY_BROADCAST", "").strip().lower()
        broadcast = flag in {"1", "true", "yes", "on", "both", "all"}
    if fallback is None:
        fallback = True
    if fallback:
        if (
            os.getenv("SLACK_BOT_TOKEN")
            or os.getenv("SLACK_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK_URL")
        ):
            return FallbackNotifier()
    if broadcast:
        notifiers: list[Notifier] = []
        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        if platform in {"auto", "both", "broadcast", "all"}:
            if slack_url:
                notifiers.append(Notifier(platform="slack", webhook_url=slack_url))
            if discord_url:
                notifiers.append(Notifier(platform="discord", webhook_url=discord_url))
        else:
            if platform == "slack" and slack_url:
                notifiers.append(Notifier(platform="slack", webhook_url=slack_url))
            if platform == "discord" and discord_url:
                notifiers.append(Notifier(platform="discord", webhook_url=discord_url))
        if len(notifiers) >= 2:
            return BroadcastNotifier(notifiers)
        if len(notifiers) == 1:
            return notifiers[0]
        return Notifier(platform=platform)
    return Notifier(platform=platform)


def get_notifiers_from_env() -> list[Notifier]:
    try:
        if (
            os.getenv("SLACK_BOT_TOKEN")
            or os.getenv("SLACK_WEBHOOK_URL")
            or os.getenv("DISCORD_WEBHOOK_URL")
        ):
            return [FallbackNotifier()]
    except Exception:
        pass
    return [Notifier(platform="auto")]
