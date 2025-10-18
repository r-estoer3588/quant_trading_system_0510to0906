"""通知ユーティリティ。

- Slack Web API / Discord Webhook に対応
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

# Ensure .env is loaded early so env vars are available even if settings is not imported yet
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    if load_dotenv is not None:
        _ROOT = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=_ROOT / ".env", override=False)  # does nothing if missing
except Exception:
    pass

try:  # pragma: no cover - optional dependency
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except Exception:  # pragma: no cover - missing optional dependency
    WebClient = None
    SlackApiError = Exception

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
    "SimpleSlackNotifier",
    "RichSlackNotifier",
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
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=_JST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M JST")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("notifier")
    if logger.handlers:
        return logger
    # 環境変数 NOTIFIER_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)
    _lvl = os.getenv("NOTIFIER_LOG_LEVEL", "INFO").strip().upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(_lvl, logging.INFO))
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


def format_table(rows: list[Iterable[Any]], headers: list[str] | None = None, max_width: int = 80) -> str:
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


def chunk_fields(name: str, items: list[str], inline: bool = True, max_per_field: int = 15) -> list[dict[str, Any]]:
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
    if os.getenv("SLACK_BOT_TOKEN"):
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


def _group_trades_by_side(
    trades: list[dict[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Group trades by side and compute notional sums."""
    impact_date: datetime | None = None
    include_system = any(t.get("system") for t in trades)
    groups: dict[str, dict[str, Any]] = {
        "BUY": {"rows": [], "total": 0.0},
        "SELL": {"rows": [], "total": 0.0},
    }
    for t in trades:
        sym = str(t.get("symbol"))
        side = str(t.get("action", t.get("side", ""))).upper()
        qty = int(t.get("qty", t.get("shares", 0)))
        price = float(t.get("price", t.get("entry_price", 0.0)))
        notional = qty * price
        entry_date = t.get("entry_date")
        if entry_date:
            try:
                d = datetime.fromisoformat(str(entry_date)).replace(tzinfo=_JST)
                if impact_date is None or d > impact_date:
                    impact_date = d
            except Exception:
                pass
        row: list[str] = [sym]
        if include_system:
            row.append(str(t.get("system", "")))
        row.extend([str(qty), f"{price:.2f}", f"{notional:.2f}"])
        g = groups.setdefault(side, {"rows": [], "total": 0.0})
        g["rows"].append(row)
        g["total"] += notional
    headers = ["SYMBOL"] + (["SYSTEM"] if include_system else []) + ["QTY", "PRICE", "AMOUNT"]
    for g in groups.values():
        g["headers"] = headers
    impact_str = impact_date.date().isoformat() if impact_date else datetime.now(tz=_JST).date().isoformat()
    return impact_str, groups


class Notifier:
    def __init__(self, platform: str = "auto", webhook_url: str | None = None):
        if platform == "auto":
            platform = detect_default_platform()
        self.platform = platform
        if platform == "slack":
            # Slack は Webhook を使用せず Web API のみサポート
            self.webhook_url = None
        elif platform == "discord":
            self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        else:
            self.webhook_url = webhook_url
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
                    self.logger.error("Slack API送信に失敗しました（Webhookへはフォールバックしません）")
                return
        if not self.webhook_url:
            self.logger.warning("webhook 未設定のため送信をスキップします platform=%s", self.platform)
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
            or os.getenv("SLACK_CHANNEL_LOGS", "").strip()
            or os.getenv("SLACK_CHANNEL_SIGNALS", "").strip()
            or os.getenv("SLACK_CHANNEL_EQUITY", "").strip()
        )
        # 前提条件チェック（詳細な診断をログ出力）
        missing: list[str] = []
        if WebClient is None:
            missing.append("slack_sdk 未インストール（pip install slack_sdk）")
        if not token:
            missing.append("SLACK_BOT_TOKEN 未設定")
        if not channel:
            missing.append("送信先チャンネル未設定（payload._channel / SLACK_CHANNEL / SLACK_CHANNEL_ID）")
        if missing:
            self.logger.warning("Slack API 前提条件不足: %s", ", ".join(missing))
            return False

        blocks = payload.get("blocks")
        text = payload.get("text") or "Notification"
        try:  # pragma: no cover
            client = WebClient(token=token)
            client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks,
            )
            self.logger.info("sent via Slack Web API to channel=%s", channel)
            return True
        except SlackApiError as e:
            # エラーメッセージを詳細化（チャネル含む）
            resp = getattr(e, "response", None)
            try:
                msg = resp.get("error") if resp else str(e)
            except Exception:
                msg = str(e)
            self.logger.warning("Slack API error on channel=%s: %s", channel, truncate(msg, 300))
            return False
        except Exception as e:  # pragma: no cover
            self.logger.warning("Slack API exception on channel=%s: %s", channel, e)
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
                        {
                            "name": truncate(k, 256),
                            "value": truncate(str(v), 1024),
                            "inline": True,
                        }
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
                    "text": {
                        "type": "mrkdwn",
                        "text": truncate(f"*{title}*\n{desc}", 3000),
                    },
                }
            ]
            if isinstance(fields, dict):
                text = "\n".join(f"*{k}*: {v}" for k, v in fields.items())
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": truncate(text, 3000)},
                    }
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
            (0 if not fields else (len(fields) if isinstance(fields, list) else len(fields))),
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
                tag = "<!channel>" if str(mention).lower() in {"channel", "@everyone"} else "<!here>"
                desc = f"{tag}\n" + desc
            else:
                content = "@everyone" if str(mention).lower() in {"channel", "@everyone"} else "@here"

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
                        {
                            "name": truncate(k, 256),
                            "value": truncate(str(v), 1024),
                            "inline": True,
                        }
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
                    "text": {
                        "type": "mrkdwn",
                        "text": truncate(f"*{title}*\n{desc}", 3000),
                    },
                }
            ]
            if isinstance(fields, dict):
                text = "\n".join(f"*{k}*: {v}" for k, v in fields.items())
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": truncate(text, 3000)},
                    }
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
            (0 if not fields else (len(fields) if isinstance(fields, list) else len(fields))),
            bool(image_url),
        )
        if channel:
            payload["_channel"] = channel
        self._post(payload)

    def send_signals(self, system_name: str, signals: list[str], *, channel: str | None = None) -> None:
        direction = SYSTEM_POSITION.get(system_name.lower(), "")
        color = COLOR_LONG if direction == "long" else COLOR_SHORT if direction == "short" else COLOR_NEUTRAL
        title = f"📢 {system_name} 日次シグナル ・ {now_jst_str()}"
        ch = channel or (os.getenv("SLACK_CHANNEL_SIGNALS") if self.platform == "slack" else None)
        if not signals:
            self.send(title, "本日のシグナルはありません", color=color, channel=ch)
            self.logger.info("signals %s direction=%s count=0", system_name, direction or "none")
            return
        emoji = "🟢" if direction == "long" else ("🔴" if direction == "short" else "")
        items = [f"{emoji} {s}" if emoji else s for s in signals]
        fields = chunk_fields("銘柄", items, inline=False)
        preview = ", ".join(signals[:10])
        if len(signals) > 10:
            preview += " ..."
        summary = f"シグナル数: {len(signals)}\n{preview}" if preview else f"シグナル数: {len(signals)}"
        self.send(title, summary, fields=fields, color=color, channel=ch)
        self.logger.info(
            "signals %s direction=%s count=%d",
            system_name,
            direction or "none",
            len(signals),
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
        period_with_run = f"{period}, 実行日 ・ {now_jst_str()}" if period else f"実行日 ・ {now_jst_str()}"
        self.send_backtest_ex(system_name, period_with_run, stats, ranking, channel=channel)
        summary = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:3])
        self.logger.info("backtest %s stats=%s top=%d", system_name, summary, min(len(ranking), 10))

    def send_trade_report(self, system_name: str, trades: list[dict[str, Any]]) -> None:
        impact, groups = _group_trades_by_side(trades)
        if not any(g["rows"] for g in groups.values()):
            title = f"🧾 {system_name} 売買結果 ・ {impact}"
            self.send(title, "本日の売買はありません")
            self.logger.info("trade report %s count=0", system_name)
            return
        for side in ("BUY", "SELL"):
            g = groups.get(side)
            if not g or not g["rows"]:
                continue
            title = f"🧾 {system_name} {side} 注文 ・ {impact}"
            table = format_table(g["rows"], headers=g["headers"])
            self.send(title, table)
            self.logger.info(
                "trade report %s side=%s count=%d notional=%.2f",
                system_name,
                side.lower(),
                len(g["rows"]),
                g["total"],
            )

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
    ) -> None:
        title = f"📊 {system_name} {period_type} サマリー ・ {period_label}, 実行日 ・ {now_jst_str()}"
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
        color = COLOR_LONG if direction == "long" else COLOR_SHORT if direction == "short" else COLOR_NEUTRAL
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
            "backtest_ex %s stats=%s top=%d",
            system_name,
            summary,
            min(len(ranking), 10),
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


class SimpleSlackNotifier(Notifier):
    """(New) 単体 Slack API 専用ノーティファ。

    目的:
        以前 `FallbackNotifier` が担っていた "Slack API だけで送る" 用途を簡素化。
        Webhook へのフォールバックや Discord 連鎖など複合挙動を排した最小形。

    注意:
        - Bot Token/Channel が未設定なら send 系は警告して終了。
        - 後方互換: 旧 FallbackNotifier ログプレフィックス "fallback:" は廃止し
          視認性向上のため "slack_api:" に変更。
    """

    def __init__(self) -> None:  # noqa: D401
        super().__init__(platform="slack")
        self._slack_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        self._slack_default_ch = os.getenv("SLACK_CHANNEL", "").strip() or os.getenv("SLACK_CHANNEL_ID", "").strip()
        # 追加フォールバック: 一般チャンネル指定が無い場合、ログ/シグナル/エクイティ用が一つでもあればその最初をデフォルトに採用
        # これによりユーザが SLACK_CHANNEL_* 系のみ .env に設定しているケースでも debug / send() が沈黙しない。
        if not self._slack_default_ch:
            for _alt_env in [
                "SLACK_CHANNEL_LOGS",
                "SLACK_CHANNEL_SIGNALS",
                "SLACK_CHANNEL_EQUITY",
            ]:
                _v = os.getenv(_alt_env, "").strip()
                if _v:
                    self._slack_default_ch = _v
                    break
        # 役割別チャンネル (存在しないものは空文字)
        self._ch_logs = os.getenv("SLACK_CHANNEL_LOGS", "").strip()
        self._ch_signals = os.getenv("SLACK_CHANNEL_SIGNALS", "").strip()
        self._ch_equity = os.getenv("SLACK_CHANNEL_EQUITY", "").strip()

    def _resolve_channel(self, kind: str | None, explicit: str | None = None) -> str | None:
        """役割 (kind) と明示指定 explicit から最終チャンネルを決定。

        kind:
            logs     -> SLACK_CHANNEL_LOGS
            signals  -> SLACK_CHANNEL_SIGNALS
            equity   -> SLACK_CHANNEL_EQUITY
            generic/None -> self._slack_default_ch
        explicit があればそれを最優先。
        """
        if explicit:
            return explicit
        if kind == "logs" and self._ch_logs:
            return self._ch_logs
        if kind == "signals" and self._ch_signals:
            return self._ch_signals
        if kind == "equity" and self._ch_equity:
            return self._ch_equity
        return self._slack_default_ch or self._ch_logs or self._ch_signals or self._ch_equity

    # 旧 FallbackNotifier 互換 private メソッド名を保持
    def _slack_send_text(
        self,
        text: str,
        *,
        channel: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> bool:  # noqa: D401
        debug_mode = os.getenv("SLACK_DEBUG_VERBOSE") == "1"
        run_id = os.getenv("BACKTEST_RUN_ID", "")
        # 失敗理由を最後にまとめて表示するためのバッファ
        debug_reasons: list[str] = []

        if _notifications_disabled():
            msg = "slack_api: 通知送信は無効化されています（テスト/CI/環境変数）"
            self.logger.info(msg)
            if debug_mode:
                print(f"[SLACK_DEBUG][run_id={run_id}] {msg}")
            return True
        ch = channel or self._slack_default_ch
        if not ch:
            reason = "channel_not_set"
            self.logger.warning("slack_api: チャンネル未設定のため送信スキップ")
            if debug_mode:
                debug_reasons.append(reason)
                print(
                    f"[SLACK_DEBUG][run_id={run_id}] failed reason={reason} token_set={bool(self._slack_token)} text_len={len(text)}"
                )
            return False
        token = self._slack_token
        if token and WebClient is not None:
            try:  # pragma: no cover
                client = WebClient(token=token)
                client.chat_postMessage(channel=ch, text=text, blocks=blocks)
                self.logger.info("slack_api: sent to %s", ch)
                if debug_mode:
                    print(
                        f"[SLACK_DEBUG][run_id={run_id}] success channel={ch} text_len={len(text)} blocks={bool(blocks)}"
                    )
                return True
            except SlackApiError as e:
                resp = getattr(e, "response", None)
                try:
                    msg = resp.get("error") if resp else str(e)
                except Exception:
                    msg = str(e)
                self.logger.warning("slack_api: error %s", truncate(msg, 200))
                if debug_mode:
                    debug_reasons.append(f"slack_api_error:{msg}")
            except Exception as e:  # pragma: no cover
                self.logger.warning("slack_api: exception %s", e)
                if debug_mode:
                    debug_reasons.append(f"exception:{type(e).__name__}:{e}")
        else:
            if debug_mode:
                debug_reasons.append(
                    f"client_unavailable token_set={bool(token)} webclient={'yes' if WebClient is not None else 'no'}"
                )
        if debug_mode:
            # ここまで到達 = 失敗
            print(f"[SLACK_DEBUG][run_id={run_id}] failed channel={ch} reasons={';'.join(debug_reasons) or 'unknown'}")
        return False

    # 代表的シグナル類をシンプル送信（失敗しても例外化せずログのみ）
    def send(self, title: str, message: str, *_, **__) -> None:  # type: ignore[override]
        # 汎用メッセージは logs チャンネルへ
        text = f"{title}\n{message}" if message else title
        ch = self._resolve_channel("logs", None)
        self._slack_send_text(text, channel=ch)

    def send_signals(self, system_name: str, signals: list[str], *, channel: str | None = None) -> None:  # type: ignore[override]
        preview = ", ".join(signals[:10]) + (" ..." if len(signals) > 10 else "") if signals else "(none)"
        text = f"📢 {system_name} Signals {now_jst_str()}\ncount={len(signals)}\n{preview}"  # noqa: E501
        ch = self._resolve_channel("signals", channel)
        self._slack_send_text(text, channel=ch)

    def send_backtest(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[str],
        *,
        channel: str | None = None,
    ) -> None:  # type: ignore[override]
        summary = ", ".join(f"{k}={v}" for k, v in list(stats.items())[:5])
        text = f"📊 {system_name} Backtest {period} {now_jst_str()}\n{summary}"
        ch = self._resolve_channel("logs", channel)
        self._slack_send_text(text, channel=ch)

    def send_backtest_ex(self, *args, **kwargs) -> None:  # type: ignore[override]
        self.send_backtest(*args, **kwargs)

    def send_trade_report(self, system_name: str, trades: list[dict[str, Any]]) -> None:  # type: ignore[override]
        text = f"🧾 {system_name} Trades {now_jst_str()} count={len(trades)}"
        ch = self._resolve_channel("logs", None)
        self._slack_send_text(text, channel=ch)

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
    ) -> None:  # type: ignore[override]
        kv = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:10])
        text = f"📊 {system_name} {period_type} {period_label} {now_jst_str()}\n{kv}"
        ch = self._resolve_channel("logs", None)
        self._slack_send_text(text, channel=ch)


class FallbackNotifier(SimpleSlackNotifier):  # type: ignore
    """(Deprecated) 互換ラッパー。

    旧クラス名への直接参照を残したままでも挙動は SimpleSlackNotifier に委譲する。
    使用時に一度だけ WARNING を出す。
    """

    _warned = False

    def __init__(self) -> None:  # noqa: D401
        if not FallbackNotifier._warned:
            logging.getLogger("notifier").warning(
                "FallbackNotifier は非推奨です。SimpleSlackNotifier へ移行してください。"
            )
            FallbackNotifier._warned = True
        super().__init__()

    def _slack_send_text(
        self,
        text: str,
        *,
        channel: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> bool:
        if _notifications_disabled():
            self.logger.info("通知送信は無効化されています（テスト/CI/環境変数）")
            return True
        ch = channel or self._slack_default_ch
        if not ch:
            return False
        token = self._slack_token
        if token and WebClient is not None:
            try:  # pragma: no cover
                client = WebClient(token=token)
                client.chat_postMessage(channel=ch, text=text, blocks=blocks)
                self.logger.info("fallback: sent via Slack API to %s", ch)
                return True
            except SlackApiError as e:
                resp = getattr(e, "response", None)
                try:
                    msg = resp.get("error") if resp else str(e)
                except Exception:
                    msg = str(e)
                self.logger.warning("fallback: Slack API error: %s", truncate(msg, 200))
            except Exception as e:
                self.logger.warning("fallback: Slack API exception: %s", e)
        return False

    def _slack_upload_file(self, image_path: str, *, title: str, initial_comment: str, channel: str | None) -> bool:
        if _notifications_disabled():
            return True
        token = self._slack_token
        ch = channel or self._slack_default_ch
        if not token or not ch or WebClient is None:
            return False
        try:  # pragma: no cover
            client = WebClient(token=token)
            client.files_upload_v2(
                channel=ch,
                initial_comment=initial_comment,
                title=title,
                file=image_path,
            )
            self.logger.info("fallback: file uploaded via Slack API to %s", ch)
            return True
        except SlackApiError as e:
            resp = getattr(e, "response", None)
            try:
                msg = resp.get("error") if resp else str(e)
            except Exception:
                msg = str(e)
            self.logger.warning("fallback: Slack file upload error: %s", truncate(msg, 200))
            return False
        except Exception as e:
            self.logger.warning("fallback: Slack file upload exception: %s", e)
            return False

    def _discord_call(self, fn_name: str, *args, **kwargs) -> bool:
        if not self._discord:
            return False
        try:
            getattr(self._discord, fn_name)(*args, **kwargs)
            self.logger.info("fallback: sent via Discord (%s)", fn_name)
            return True
        except Exception as e:  # pragma: no cover
            self.logger.warning("fallback: Discord send failed (%s) %s", fn_name, e)
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
        blocks: list[dict[str, Any]] | None = None
        if isinstance(fields, dict) and fields:

            def _fmt(v: Any) -> str:
                try:
                    if isinstance(v, (int | float)):
                        return f"{float(v):.2f}"
                    # 数値文字列も丸めを試行
                    _f = float(str(v))
                    return f"{_f:.2f}"
                except Exception:
                    return str(v)

            kv = ", ".join(f"{k}={_fmt(v)}" for k, v in list(fields.items())[:10])
            lines.append(kv)
        elif isinstance(fields, list) and fields:
            blocks = []
            for f in fields:
                name = str(f.get("name", ""))
                value = str(f.get("value", ""))
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*{name}*\n{value}"},
                    }
                )
        text = "\n".join(lines)
        ch = channel or os.getenv("SLACK_CHANNEL_LOGS") or None
        if self._slack_send_text(text, channel=ch, blocks=blocks):
            return
        if not self._discord_call("send", title, message, fields=fields, image_url=image_url, color=color):
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
        if image_path and self._slack_upload_file(image_path, title=title, initial_comment=text, channel=ch):
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

    def send_signals(self, system_name: str, signals: list[str], *, channel: str | None = None) -> None:
        direction = SYSTEM_POSITION.get(system_name.lower(), "")
        title = f"📢 {system_name} 日次シグナル ・ {now_jst_str()}"
        ch = channel or os.getenv("SLACK_CHANNEL_SIGNALS") or None
        if not signals:
            text = f"{title}\n本日のシグナルはありません"
            if self._slack_send_text(text, channel=ch):
                return
            if not self._discord_call("send_signals", system_name, signals):
                raise RuntimeError("notification failed (slack+discord)")
            return

        emoji = "🟢" if direction == "long" else ("🔴" if direction == "short" else "")
        items = [f"{emoji} {s}" if emoji else str(s) for s in signals]
        fields = chunk_fields("銘柄", items, inline=False)
        preview = ", ".join(signals[:10])
        if len(signals) > 10:
            preview += " ..."
        summary = f"シグナル数: {len(signals)}\n{preview}" if preview else f"シグナル数: {len(signals)}"
        blocks: list[dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{title}*\n{summary}"},
            }
        ]
        for f in fields:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{f['name']}*\n{f['value']}"},
                }
            )
        if self._slack_send_text(summary, channel=ch, blocks=blocks):
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
        period_with_run = f"{period}, 実行日 ・ {now_jst_str()}" if period else f"実行日 ・ {now_jst_str()}"
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
        if image_path and self._slack_upload_file(image_path, title=title, initial_comment=text, channel=ch):
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
        impact, groups = _group_trades_by_side(trades)
        if not any(g["rows"] for g in groups.values()):
            text = f"🧾 {system_name} 売買結果 ・ {impact}\n本日の売買はありません"
            if self._slack_send_text(text):
                return
            if not self._discord_call("send_trade_report", system_name, []):
                raise RuntimeError("notification failed (slack+discord)")
            return
        for side in ("BUY", "SELL"):
            g = groups.get(side)
            if not g or not g["rows"]:
                continue
            title = f"🧾 {system_name} {side} 注文 ・ {impact}"
            table = format_table(g["rows"], headers=g["headers"])
            text = f"{title}\n{table}"
            if self._slack_send_text(text):
                continue
            side_trades = [t for t in trades if str(t.get("action", t.get("side", ""))).upper() == side]
            if not self._discord_call("send_trade_report", system_name, side_trades):
                raise RuntimeError("notification failed (slack+discord)")

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
    ) -> None:  # noqa: E501
        title = f"📊 {system_name} {period_type} サマリー ・ {period_label}, 実行日 ・ {now_jst_str()}"
        kv = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:10])
        text = f"{title}\n{kv}" if kv else title
        if self._slack_send_text(text):
            return
        if not self._discord_call(
            "send_summary",
            system_name,
            period_type,
            period_label,
            summary,
            image_url=image_url,
        ):  # noqa: E501
            raise RuntimeError("notification failed (slack+discord)")


def create_notifier(platform: str = "auto", broadcast: bool | None = None, fallback: bool | None = None):  # noqa: D401
    if broadcast is None:
        flag = os.getenv("NOTIFY_BROADCAST", "").strip().lower()
        broadcast = flag in {"1", "true", "yes", "on", "both", "all"}
    if fallback is None:
        fallback = True
    # Slack Rich モード判定
    use_rich = os.getenv("NOTIFY_USE_RICH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    have_token = bool(os.getenv("SLACK_BOT_TOKEN"))
    # fallback=True かつ Slack Bot Token があれば Simple/Rich Slack Notifier を優先
    if fallback and have_token:
        slack_instance: Notifier = RichSlackNotifier() if use_rich else SimpleSlackNotifier()
        if broadcast:
            notifiers: list[Notifier] = [slack_instance]
            discord_url = os.getenv("DISCORD_WEBHOOK_URL")
            if discord_url:
                notifiers.append(Notifier(platform="discord", webhook_url=discord_url))
            if len(notifiers) == 1:
                return notifiers[0]
            return BroadcastNotifier(notifiers)
        return slack_instance
    if broadcast:
        notifiers: list[Notifier] = []
        discord_url = os.getenv("DISCORD_WEBHOOK_URL")
        if platform in {"auto", "both", "broadcast", "all"}:
            if discord_url:
                notifiers.append(Notifier(platform="discord", webhook_url=discord_url))
        elif platform == "discord" and discord_url:
            notifiers.append(Notifier(platform="discord", webhook_url=discord_url))
        if len(notifiers) >= 2:
            return BroadcastNotifier(notifiers)
        if len(notifiers) == 1:
            return notifiers[0]
        return Notifier(platform=platform)
    return Notifier(platform=platform)


def get_notifiers_from_env() -> list[Notifier]:  # noqa: D401
    try:
        if os.getenv("SLACK_BOT_TOKEN"):
            if os.getenv("NOTIFY_USE_RICH", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return [RichSlackNotifier()]
            return [SimpleSlackNotifier()]
    except Exception:
        pass
    return [Notifier(platform="auto")]


class RichSlackNotifier(SimpleSlackNotifier):
    """Slack Block Kit を使いリッチ表示を行う Notifier。

    特徴:
        - ranking をトップ10まで番号付きリストで Block Kit section に分割
        - 大量の fields/stats は 3 カラム風書式を意識し複数セクションに分割
        - 旧 SimpleSlackNotifier の send_* API を互換維持
    """

    MAX_SECTION_LEN = 2800  # default Slack safety margin

    def __init__(self) -> None:  # noqa: D401
        super().__init__()
        # 環境変数でセクション長調整
        try:
            override = os.getenv("NOTIFY_RICH_MAX_SECTION")
            if override:
                v = int(override)
                if 500 < v < 3900:
                    self.MAX_SECTION_LEN = v
        except Exception:
            pass

    # 画像アップロード (files_upload_v2) を最小実装
    def _upload_image(self, image_path: str, title: str, channel: str | None) -> None:
        if not image_path or not os.path.exists(image_path):
            return
        if _notifications_disabled():
            return
        token = os.getenv("SLACK_BOT_TOKEN", "").strip()
        ch = channel or self._slack_default_ch
        if not (token and ch and WebClient is not None):
            return
        try:  # pragma: no cover
            client = WebClient(token=token)
            client.files_upload_v2(channel=ch, title=title[:80], file=image_path)
            self.logger.info("slack_api: image uploaded path=%s", image_path)
            os.environ["LAST_IMAGE_UPLOAD_OK"] = "1"
        except Exception as e:  # pragma: no cover
            self.logger.warning("slack_api: image upload failed %s", e)
            os.environ["LAST_IMAGE_UPLOAD_OK"] = "0"

    def _post_blocks(self, title: str, lines: list[str], channel: str | None = None) -> None:
        body = []
        chunk: list[str] = []
        size = 0
        for ln in lines:
            ln2 = ln if len(ln) < 4000 else ln[:3990] + "…"
            if size + len(ln2) + 1 > self.MAX_SECTION_LEN and chunk:
                body.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "\n".join(chunk)},
                    }
                )
                chunk = [ln2]
                size = len(ln2) + 1
            else:
                chunk.append(ln2)
                size += len(ln2) + 1
        if chunk:
            body.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "\n".join(chunk)},
                }
            )
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title[:150], "emoji": True},
            },
        ] + body
        self._slack_send_text(title, blocks=blocks, channel=channel)

    def send_backtest(
        self,
        system_name: str,
        period: str,
        stats: dict[str, Any],
        ranking: list[str],
        *,
        channel: str | None = None,
        image_path: str | None = None,
    ) -> None:  # type: ignore[override]
        run_id = os.getenv("BACKTEST_RUN_ID") or "-"
        title = f"📊 {system_name} Backtest {period} • {run_id}" if period else f"📊 {system_name} Backtest • {run_id}"
        stat_items = list(stats.items())
        stat_lines: list[str] = []
        for k, v in stat_items[:20]:
            stat_lines.append(f"• *{k}*: {v}")
        rank_lines: list[list[str]] = []
        if ranking:
            formatted: list[str] = []
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            for i, r in enumerate(ranking[:10], 1):
                try:
                    if isinstance(r, dict):
                        sym = r.get("symbol") or r.get("sym") or r.get("ticker") or "?"
                        extra = []
                        if "roc" in r:
                            extra.append(f"ROC:{float(r['roc']):.2f}")
                        if "volume" in r:
                            extra.append(f"Vol:{int(float(r['volume'])):,}")
                        medal = medals.get(i, "•")
                        formatted.append(f"{medal} {i}. {sym} {' '.join(extra)}")
                    else:
                        medal = medals.get(i, "•")
                        formatted.append(f"{medal} {i}. {r}")
                except Exception:
                    medal = medals.get(i, "•")
                    formatted.append(f"{medal} {i}. {r}")
            # 2 カラム整形（幅計算は現在未使用のため省略）
            left = formatted[::2]
            right = formatted[1::2]
            for idx in range(len(left)):
                left_text = left[idx]
                r = right[idx] if idx < len(right) else ""
                rank_lines.append([left_text, r])
        lines = []
        if stat_lines:
            lines.append("*Stats*\n" + "\n".join(stat_lines))
        if rank_lines:
            # 2 カラムをコードブロック化（等幅）
            table_lines = []
            for row in rank_lines:
                if row[1]:
                    table_lines.append(f"{row[0]:<25}  {row[1]}")
                else:
                    table_lines.append(row[0])
            lines.append("*Ranking (Top10)*\n```\n" + "\n".join(table_lines) + "\n```")
        if not lines:
            lines = ["(no data)"]
        self._post_blocks(title, lines, channel=channel)
        if image_path:
            self._upload_image(image_path, title=title, channel=channel)
            if os.getenv("LAST_IMAGE_UPLOAD_OK") == "0":
                # 失敗通知を追加
                self._post_blocks(
                    title + " (image upload failed)",
                    ["画像アップロードに失敗しました"],
                    channel=channel,
                )

    def send_backtest_ex(self, *args, **kwargs) -> None:  # type: ignore[override]
        self.send_backtest(*args, **kwargs)

    def send_signals(
        self,
        system_name: str,
        signals: list[str] | list[dict[str, Any]],
        *,
        channel: str | None = None,
        image_path: str | None = None,
    ) -> None:  # type: ignore[override]
        run_id = os.getenv("BACKTEST_RUN_ID") or "-"
        title = f"📢 {system_name} Signals • {run_id}"
        lines: list[str] = []
        lines.append(f"count={len(signals)} {now_jst_str()}")
        if signals:
            sample = signals[:60]
            # dict を含む場合は volume / score を取得
            has_meta = any(isinstance(x, dict) for x in sample)
            if has_meta:
                # 正規化: dict -> {'symbol':..., 'volume':..., 'score':...}
                norm = []
                for x in sample:
                    if isinstance(x, dict):
                        sym = x.get("symbol") or x.get("sym") or x.get("ticker") or "?"
                        vol = x.get("volume") or x.get("vol")
                        score = x.get("score") or x.get("roc") or x.get("rank_score")
                        norm.append((sym, vol, score))
                    else:
                        norm.append((str(x), None, None))
                # テーブル文字列化
                header = ["SYMBOL", "VOLUME", "SCORE"]
                rows_txt = []
                rows_txt.append("  ".join(f"{h:<10}" for h in header))
                for sym, vol, score in norm:
                    vtxt = f"{int(vol):,}" if isinstance(vol, (int, float)) else "-"
                    stxt = f"{float(score):.2f}" if isinstance(score, (int, float, float)) else "-"
                    rows_txt.append(f"{sym[:10]:<10}  {vtxt:<10}  {stxt:<10}")
                table = "```\n" + "\n".join(rows_txt) + "\n```"
                lines.append("*Signals*\n" + table)
            else:
                # シンボルのみ 3 カラム
                col = 3
                rows = []
                for i in range(0, len(sample), col):
                    seg = sample[i : i + col]
                    row = []
                    for s in seg:
                        row.append(f"{s[:10]:<10}")
                    while len(row) < col:
                        row.append("")
                    rows.append("  ".join(row))
                table = "```\n" + "\n".join(rows) + "\n```"
                lines.append("*Symbols*\n" + table)
        self._post_blocks(title, lines, channel=channel)
        if image_path:
            self._upload_image(image_path, title=title, channel=channel)
            if os.getenv("LAST_IMAGE_UPLOAD_OK") == "0":
                self._post_blocks(
                    title + " (image upload failed)",
                    ["画像アップロードに失敗しました"],
                    channel=channel,
                )

    def send_summary(
        self,
        system_name: str,
        period_type: str,
        period_label: str,
        summary: dict[str, Any],
        image_url: str | None = None,
        image_path: str | None = None,
    ) -> None:  # type: ignore[override]
        run_id = os.getenv("BACKTEST_RUN_ID") or "-"
        title = f"📊 {system_name} {period_type} {period_label} • {run_id}".strip()
        pairs = [f"• {k}: {v}" for k, v in list(summary.items())[:30]] or ["(empty)"]
        lines = [now_jst_str(), "*Summary*\n" + "\n".join(pairs)]
        self._post_blocks(title, lines)
        if image_path:
            self._upload_image(image_path, title=title, channel=None)
            if os.getenv("LAST_IMAGE_UPLOAD_OK") == "0":
                self._post_blocks(
                    title + " (image upload failed)",
                    ["画像アップロードに失敗しました"],
                    channel=None,
                )

    def send_trade_report(self, system_name: str, trades: list[dict[str, Any]], image_path: str | None = None) -> None:  # type: ignore[override]
        run_id = os.getenv("BACKTEST_RUN_ID") or "-"
        title = f"🧾 {system_name} Trades • {run_id}"
        lines = [f"count={len(trades)} {now_jst_str()}"]
        sample = trades[:20]
        for i, t in enumerate(sample, 1):
            try:
                sym = t.get("symbol") or t.get("sym") or "?"
                side = str(t.get("action", t.get("side", ""))).upper()
                qty = t.get("qty", t.get("shares", ""))
                price = t.get("price", t.get("entry_price", ""))
                lines.append(f"{i}. {side} {sym} {qty}@{price}")
            except Exception:
                continue
        self._post_blocks(title, lines)
        if image_path:
            self._upload_image(image_path, title=title, channel=None)
            if os.getenv("LAST_IMAGE_UPLOAD_OK") == "0":
                self._post_blocks(
                    title + " (image upload failed)",
                    ["画像アップロードに失敗しました"],
                    channel=None,
                )
