from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from dotenv import load_dotenv

load_dotenv()

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

channels = {
    "logs": os.getenv("SLACK_CHANNEL_LOGS"),
    "equity": os.getenv("SLACK_CHANNEL_EQUITY"),
    "signals": os.getenv("SLACK_CHANNEL_SIGNALS"),
}

for name, channel_id in channels.items():
    if channel_id is None:
        print(f"Error: Environment variable for '{name}' channel is missing.")
        continue
    try:
        response = client.chat_postMessage(
            channel=channel_id, text=f"✅ テスト通知: {name} チャンネルに送信しました！"
        )
        print(f"Success → {name}: ts={response['ts']}")
    except SlackApiError as e:
        print(f"Error in {name}: {e.response['error']}")
