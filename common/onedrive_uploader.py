"""OneDrive Integration with Shareable Links via Microsoft Graph API.

This module uploads files to OneDrive and generates shareable web links.

Setup:
1. Register an app at https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps
2. Set Application (client) ID in .env as ONEDRIVE_CLIENT_ID
3. Enable "Mobile and desktop applications" redirect URI: https://login.microsoftonline.com/common/oauth2/nativeclient
4. Grant API permissions: Files.ReadWrite, Files.ReadWrite.All

First-time use requires browser authentication. Token is cached locally.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# MSAL for auth
try:
    import msal
    import requests
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False
    logger.debug("msal not installed. Run: pip install msal requests")


ROOT = Path(__file__).resolve().parents[1]
TOKEN_CACHE_PATH = ROOT / "data" / "onedrive_token_cache.json"

# Microsoft Graph API endpoints
GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"
SCOPES = ["Files.ReadWrite", "Files.ReadWrite.All"]


def get_onedrive_path() -> Path | None:
    """Find OneDrive local sync folder."""
    user_home = Path.home()
    possible_paths = [
        user_home / "OneDrive",
        user_home / "OneDrive - Personal",
    ]

    custom_path = os.getenv("ONEDRIVE_PATH")
    if custom_path:
        possible_paths.insert(0, Path(custom_path))

    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    return None


def _get_msal_app():
    """Create MSAL public client application."""
    client_id = os.getenv("ONEDRIVE_CLIENT_ID")
    if not client_id:
        raise ValueError(
            "ONEDRIVE_CLIENT_ID not set. "
            "Register app at Azure Portal and add to .env"
        )

    # Load token cache
    cache = msal.SerializableTokenCache()
    if TOKEN_CACHE_PATH.exists():
        try:
            cache.deserialize(TOKEN_CACHE_PATH.read_text())
        except Exception:
            pass

    app = msal.PublicClientApplication(
        client_id,
        authority="https://login.microsoftonline.com/consumers",
        token_cache=cache,
    )

    return app, cache


def _save_cache(cache):
    """Save token cache to file."""
    if cache.has_state_changed:
        TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE_PATH.write_text(cache.serialize())


def get_access_token() -> str:
    """Get access token via MSAL (interactive auth if needed)."""
    if not MSAL_AVAILABLE:
        raise ImportError("msal not installed")

    app, cache = _get_msal_app()

    # Try to get token from cache
    accounts = app.get_accounts()
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(cache)
            return result["access_token"]

    # Interactive auth with device code flow
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise Exception(f"Failed to create device flow: {flow}")

    print("\n" + "=" * 50)
    print("🔐 OneDrive認証が必要です")
    print("=" * 50)
    print(f"\n1. ブラウザで開く: {flow['verification_uri']}")
    print(f"2. コードを入力: {flow['user_code']}")
    print("\n認証完了まで待機中...")

    result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        error = result.get("error_description", "Unknown error")
        raise Exception(f"Authentication failed: {error}")

    _save_cache(cache)
    logger.info("OneDrive authentication successful")

    return result["access_token"]


def upload_and_share(
    file_path: str | Path,
    folder_name: str = "Trading Reports",
) -> str:
    """Upload file to OneDrive and return shareable link.

    Args:
        file_path: Path to file to upload
        folder_name: Folder name in OneDrive root

    Returns:
        Shareable web URL
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }

    # Upload file to OneDrive
    upload_path = f"{folder_name}/{file_path.name}"
    upload_url = (
        f"{GRAPH_API_ENDPOINT}/me/drive/root:/{upload_path}:/content"
    )

    with open(file_path, "rb") as f:
        response = requests.put(upload_url, headers=headers, data=f)

    if response.status_code not in (200, 201):
        raise Exception(f"Upload failed: {response.status_code} {response.text}")

    file_data = response.json()
    file_id = file_data.get("id")

    # Create shareable link
    share_url = f"{GRAPH_API_ENDPOINT}/me/drive/items/{file_id}/createLink"
    share_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    share_body = {
        "type": "view",
        "scope": "anonymous",
    }

    share_response = requests.post(
        share_url, headers=share_headers, json=share_body
    )

    if share_response.status_code in (200, 201):
        link_data = share_response.json()
        web_url = link_data.get("link", {}).get("webUrl", "")
        if web_url:
            logger.info(f"Uploaded {file_path.name}: {web_url}")
            return web_url

    # Fallback to webUrl from upload response
    web_url = file_data.get("webUrl", "")
    if web_url:
        return web_url

    raise Exception("Failed to get shareable link")


def upload_to_onedrive(
    file_path: str | Path,
    folder_name: str | None = None,
) -> str:
    """Upload file to OneDrive and return shareable link.

    Falls back to local copy if API auth not configured.
    """
    folder_name = folder_name or os.getenv(
        "ONEDRIVE_REPORTS_FOLDER", "Trading Reports"
    )

    # Try API upload with shareable link
    if MSAL_AVAILABLE and os.getenv("ONEDRIVE_CLIENT_ID"):
        try:
            return upload_and_share(file_path, folder_name)
        except Exception as e:
            logger.warning(f"API upload failed, falling back to local: {e}")

    # Fallback: copy to local OneDrive folder
    file_path = Path(file_path)
    onedrive_root = get_onedrive_path()

    if not onedrive_root:
        raise FileNotFoundError("OneDrive folder not found")

    reports_dir = onedrive_root / folder_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    dest_path = reports_dir / file_path.name
    shutil.copy2(file_path, dest_path)

    logger.info(f"Copied {file_path.name} to OneDrive: {dest_path}")
    return str(dest_path)


def upload_multiple_files(
    file_paths: list[str | Path],
    folder_name: str | None = None,
) -> dict[str, str]:
    """Upload multiple files to OneDrive."""
    results = {}
    for file_path in file_paths:
        try:
            url = upload_to_onedrive(file_path, folder_name)
            results[Path(file_path).name] = url
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            results[Path(file_path).name] = f"Error: {str(e)}"
    return results


ONEDRIVE_AVAILABLE = get_onedrive_path() is not None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    onedrive = get_onedrive_path()
    print(f"OneDrive path: {onedrive}")
    print(f"MSAL available: {MSAL_AVAILABLE}")
    print(f"Client ID set: {bool(os.getenv('ONEDRIVE_CLIENT_ID'))}")
