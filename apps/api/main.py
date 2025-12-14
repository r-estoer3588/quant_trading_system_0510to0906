"""
FastAPI Backend for Alpaca Dashboard.

Provides REST API endpoints for the Next.js frontend.
Includes WebSocket for real-time signal progress updates.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing Alpaca utilities
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root
try:
    from common.broker_alpaca import get_client
    from common.position_age import days_held, load_entry_dates

    # Alpaca SDK provides these via client methods
    get_alpaca_client = get_client

    def get_positions(client):
        """Get all positions from Alpaca."""
        return client.get_all_positions()

    def get_account(client):
        """Get account info from Alpaca."""
        return client.get_account()

except ImportError as e:
    print(f"Import error: {e}")
    # Fallback mock implementations for when imports fail
    get_alpaca_client = None
    get_positions = None
    get_account = None
    days_held = None
    load_entry_dates = None


# Pydantic models for API responses
class Position(BaseModel):
    symbol: str
    qty: float
    avgEntryPrice: float
    currentPrice: float
    unrealizedPL: float
    unrealizedPLPercent: float
    holdingDays: int
    system: str


class AccountInfo(BaseModel):
    equity: float
    cash: float
    buyingPower: float
    lastEquity: float
    tradingBlocked: bool
    patternDayTrader: bool


class SystemAllocation(BaseModel):
    system: str
    totalValue: float
    positions: list[dict[str, Any]]


class DashboardData(BaseModel):
    account: AccountInfo
    positions: list[Position]
    allocations: list[SystemAllocation]
    lastUpdated: str


# Create FastAPI app
app = FastAPI(
    title="Alpaca Dashboard API",
    version="2.0.0",
    description="Backend API for the Alpaca Trading Dashboard",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_symbol_system_map() -> dict[str, str]:
    """Load symbol to system mapping from JSON file."""
    try:
        map_path = (
            Path(__file__).parent.parent.parent / "data" / "symbol_system_map.json"
        )
        if map_path.exists():
            with open(map_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_position_holding_days(symbol: str) -> int:
    """Get the number of days a position has been held."""
    if days_held is None or load_entry_dates is None:
        return 0
    try:
        entry_dates = load_entry_dates()
        entry_date = entry_dates.get(symbol.upper())
        if entry_date:
            result = days_held(entry_date)
            return result if result is not None else 0
        return 0
    except Exception:
        return 0


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Alpaca Dashboard API", "version": "2.0.0"}


@app.get("/api/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """Get all dashboard data (account, positions, allocations)."""
    try:
        # Initialize Alpaca client
        if get_alpaca_client is None:
            raise HTTPException(status_code=500, detail="Alpaca client not available")

        client = get_alpaca_client()

        # Get account info
        account = get_account(client)
        account_info = AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buyingPower=float(account.buying_power),
            lastEquity=float(account.last_equity),
            tradingBlocked=account.trading_blocked,
            patternDayTrader=account.pattern_day_trader,
        )

        # Get positions
        raw_positions = get_positions(client)
        symbol_map = load_symbol_system_map()

        positions: list[Position] = []
        system_totals: dict[str, dict] = {}

        for pos in raw_positions:
            symbol = pos.symbol
            system = symbol_map.get(symbol, "unknown")
            qty = float(pos.qty)
            avg_entry = float(pos.avg_entry_price)
            current = float(pos.current_price)
            unrealized_pl = float(pos.unrealized_pl)
            market_value = qty * current

            # Calculate percent
            cost_basis = qty * avg_entry
            pl_percent = (unrealized_pl / cost_basis * 100) if cost_basis != 0 else 0

            # Get holding days
            holding_days = get_position_holding_days(symbol)

            positions.append(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avgEntryPrice=avg_entry,
                    currentPrice=current,
                    unrealizedPL=unrealized_pl,
                    unrealizedPLPercent=pl_percent,
                    holdingDays=holding_days,
                    system=system,
                )
            )

            # Aggregate for allocations
            if system not in system_totals:
                system_totals[system] = {"totalValue": 0, "positions": []}
            system_totals[system]["totalValue"] += market_value
            system_totals[system]["positions"].append(
                {"symbol": symbol, "value": market_value}
            )

        # Build allocations
        allocations = [
            SystemAllocation(
                system=sys,
                totalValue=data["totalValue"],
                positions=data["positions"],
            )
            for sys, data in system_totals.items()
        ]

        return DashboardData(
            account=account_info,
            positions=positions,
            allocations=allocations,
            lastUpdated=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/account", response_model=AccountInfo)
async def get_account_info():
    """Get account information only."""
    if get_alpaca_client is None or get_account is None:
        raise HTTPException(status_code=500, detail="Alpaca client not available")

    try:
        client = get_alpaca_client()
        account = get_account(client)
        return AccountInfo(
            equity=float(account.equity),
            cash=float(account.cash),
            buyingPower=float(account.buying_power),
            lastEquity=float(account.last_equity),
            tradingBlocked=account.trading_blocked,
            patternDayTrader=account.pattern_day_trader,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions", response_model=list[Position])
async def get_positions_list():
    """Get positions list only."""
    if get_alpaca_client is None or get_positions is None:
        raise HTTPException(status_code=500, detail="Alpaca client not available")

    try:
        client = get_alpaca_client()
        raw_positions = get_positions(client)
        symbol_map = load_symbol_system_map()

        positions: list[Position] = []
        for pos in raw_positions:
            symbol = pos.symbol
            system = symbol_map.get(symbol, "unknown")
            qty = float(pos.qty)
            avg_entry = float(pos.avg_entry_price)
            current = float(pos.current_price)
            unrealized_pl = float(pos.unrealized_pl)
            cost_basis = qty * avg_entry
            pl_percent = (unrealized_pl / cost_basis * 100) if cost_basis != 0 else 0
            holding_days = get_position_holding_days(symbol)

            positions.append(
                Position(
                    symbol=symbol,
                    qty=qty,
                    avgEntryPrice=avg_entry,
                    currentPrice=current,
                    unrealizedPL=unrealized_pl,
                    unrealizedPLPercent=pl_percent,
                    holdingDays=holding_days,
                    system=system,
                )
            )

        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Signal Generation Endpoints (for Integrated Dashboard)
# ============================================================================


class SystemState(BaseModel):
    """State of a single trading system."""

    system: str
    target: int
    filterPass: int
    setupPass: int
    tradelist: int
    entry: int
    exit: int
    status: str


class SignalCandidate(BaseModel):
    """A signal candidate from the system."""

    symbol: str
    system: str
    side: str
    score: float
    price: float
    action: str


class IntegratedState(BaseModel):
    """Complete state of the integrated dashboard."""

    systems: list[SystemState]
    candidates: list[SignalCandidate]
    lastUpdated: str


@app.get("/api/signals/state", response_model=IntegratedState)
async def get_signal_state():
    """Get current state of all trading systems."""
    import random

    systems = []
    systems_config = [
        {"name": "System1", "type": "long"},
        {"name": "System2", "type": "short"},
        {"name": "System3", "type": "long"},
        {"name": "System4", "type": "long"},
        {"name": "System5", "type": "long"},
        {"name": "System6", "type": "short"},
        {"name": "System7", "type": "short"},
    ]

    for cfg in systems_config:
        systems.append(
            SystemState(
                system=cfg["name"],
                target=6200,
                filterPass=1000 + random.randint(0, 500),
                setupPass=50 + random.randint(0, 100),
                tradelist=random.randint(0, 10),
                entry=random.randint(0, 5),
                exit=random.randint(0, 3),
                status="idle",
            )
        )

    # Generate mock candidates
    symbols = [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "AMD",
        "CRM",
        "ORCL",
    ]
    candidates = []

    for _ in range(15):
        cfg = random.choice(systems_config)
        candidates.append(
            SignalCandidate(
                symbol=random.choice(symbols),
                system=cfg["name"],
                side=cfg["type"],
                score=random.randint(50, 100),
                price=round(100 + random.random() * 400, 2),
                action="entry" if random.random() > 0.3 else "exit",
            )
        )

    # Sort by score
    candidates.sort(key=lambda x: x.score, reverse=True)

    return IntegratedState(
        systems=systems,
        candidates=candidates,
        lastUpdated=datetime.now().isoformat(),
    )


class RunSignalsRequest(BaseModel):
    """Request to run signal generation."""

    capital: float = 100000
    longShare: int = 50
    symbolLimit: int = 500
    testMode: str = "test_symbols"  # test_symbols, mini, quick, sample, or "" for full
    skipExternal: bool = True  # Skip external API calls for faster testing


class RunSignalsResponse(BaseModel):
    """Response from signal generation run."""

    success: bool
    message: str
    systems: list[SystemState]
    candidates: list[SignalCandidate]
    executionTime: float
    lastUpdated: str


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Progress file path
PROGRESS_FILE = PROJECT_ROOT / "logs" / "progress_today.jsonl"

# Global state for tracking running pipeline
_running_process: subprocess.Popen | None = None
_run_start_time: float | None = None


class ConnectionManager:
    """Manages WebSocket connections for progress updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


async def watch_progress_file(websocket: WebSocket):
    """Watch progress_today.jsonl and send updates via WebSocket."""
    last_position = 0

    while True:
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                    for line in new_lines:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                await websocket.send_json(event)
                            except json.JSONDecodeError:
                                pass
        except Exception:
            pass

        await asyncio.sleep(0.5)


@app.websocket("/ws/signals/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket)

    try:
        # Reset progress file position
        if PROGRESS_FILE.exists():
            initial_size = PROGRESS_FILE.stat().st_size
        else:
            initial_size = 0

        last_position = initial_size

        while True:
            # Check for new progress events
            if PROGRESS_FILE.exists():
                try:
                    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        new_position = f.tell()

                        if new_position > last_position:
                            last_position = new_position
                            for line in new_lines:
                                line = line.strip()
                                if line:
                                    try:
                                        event = json.loads(line)
                                        await websocket.send_json(event)
                                    except json.JSONDecodeError:
                                        pass
                except Exception:
                    pass

            # Check if client is still connected
            try:
                # Non-blocking receive with timeout
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


def parse_results_csv() -> list[SignalCandidate]:
    """Parse the latest results CSV to extract candidates."""
    candidates = []

    # Primary location: data_cache/signals/signals_final_*.csv
    signals_dir = PROJECT_ROOT / "data_cache" / "signals"

    if signals_dir.exists():
        csv_files = list(signals_dir.glob("signals_final_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            candidates = _parse_signals_csv(latest_csv)
            if candidates:
                return candidates

    # Fallback: results_csv/final_allocation_*.csv
    results_dir = PROJECT_ROOT / "results_csv"
    if results_dir.exists():
        csv_files = list(results_dir.glob("final_allocation_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            candidates = _parse_signals_csv(latest_csv)

    return candidates


def _parse_signals_csv(csv_path: Path) -> list[SignalCandidate]:
    """Parse a signals CSV file and return candidates."""
    candidates = []
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Handle different column name conventions
        symbol_col = next(
            (c for c in ["symbol", "Symbol", "ticker", "Ticker"] if c in df.columns),
            None,
        )
        system_col = next((c for c in ["system", "System"] if c in df.columns), None)
        side_col = next(
            (c for c in ["side", "Side", "direction"] if c in df.columns), None
        )
        score_col = next(
            (c for c in ["score", "Score", "rank_value"] if c in df.columns), None
        )
        price_col = next(
            (c for c in ["entry_price", "price", "close", "Close"] if c in df.columns),
            None,
        )
        action_col = next(
            (c for c in ["signal_type", "action", "type"] if c in df.columns), None
        )

        for _, row in df.iterrows():
            candidates.append(
                SignalCandidate(
                    symbol=str(row.get(symbol_col, "")) if symbol_col else "",
                    system=str(row.get(system_col, "")) if system_col else "",
                    side=str(row.get(side_col, "long")) if side_col else "long",
                    score=float(row.get(score_col, 0)) if score_col else 0,
                    price=float(row.get(price_col, 0)) if price_col else 0,
                    action=str(row.get(action_col, "entry")) if action_col else "entry",
                )
            )
    except Exception:
        pass

    return candidates


@app.post("/api/signals/run", response_model=RunSignalsResponse)
async def run_signals(request: RunSignalsRequest):
    """Run signal generation for all systems (REAL EXECUTION)."""
    global _running_process, _run_start_time

    start_time = time.time()
    _run_start_time = start_time

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "scripts.run_all_systems_today",
        "--save-csv",
    ]

    # Add capital allocation
    capital_long = request.capital * request.longShare / 100
    capital_short = request.capital * (100 - request.longShare) / 100

    if capital_long > 0:
        cmd.extend(["--capital-long", str(int(capital_long))])
    if capital_short > 0:
        cmd.extend(["--capital-short", str(int(capital_short))])

    # Add test mode for safety
    if request.testMode:
        cmd.extend(["--test-mode", request.testMode])

    # Skip external API calls for faster testing
    if request.skipExternal:
        cmd.append("--skip-external")

    # Add benchmark flag for timing info
    cmd.append("--benchmark")

    # Clear progress file before starting
    try:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.touch()
    except Exception:
        pass

    systems_config = [
        {"name": "System1", "type": "long"},
        {"name": "System2", "type": "short"},
        {"name": "System3", "type": "long"},
        {"name": "System4", "type": "long"},
        {"name": "System5", "type": "long"},
        {"name": "System6", "type": "short"},
        {"name": "System7", "type": "short"},
    ]

    try:
        # Run subprocess asynchronously
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT),
            timeout=600,  # 10 minute timeout
        )

        success = result.returncode == 0
        message = (
            "Signal generation complete."
            if success
            else f"Pipeline error: {result.stderr[:200]}"
        )

        # Parse results
        candidates = parse_results_csv()

        # Build system states from progress file
        systems = []
        for cfg in systems_config:
            systems.append(
                SystemState(
                    system=cfg["name"],
                    target=6200,
                    filterPass=0,
                    setupPass=0,
                    tradelist=len([c for c in candidates if c.system == cfg["name"]]),
                    entry=len(
                        [
                            c
                            for c in candidates
                            if c.system == cfg["name"] and c.action == "entry"
                        ]
                    ),
                    exit=len(
                        [
                            c
                            for c in candidates
                            if c.system == cfg["name"] and c.action == "exit"
                        ]
                    ),
                    status="complete" if success else "error",
                )
            )

    except subprocess.TimeoutExpired:
        success = False
        message = "Pipeline timed out after 10 minutes"
        candidates = []
        systems = [
            SystemState(
                system=cfg["name"],
                target=0,
                filterPass=0,
                setupPass=0,
                tradelist=0,
                entry=0,
                exit=0,
                status="error",
            )
            for cfg in systems_config
        ]
    except Exception as e:
        success = False
        message = f"Error running pipeline: {str(e)}"
        candidates = []
        systems = [
            SystemState(
                system=cfg["name"],
                target=0,
                filterPass=0,
                setupPass=0,
                tradelist=0,
                entry=0,
                exit=0,
                status="error",
            )
            for cfg in systems_config
        ]

    execution_time = time.time() - start_time

    return RunSignalsResponse(
        success=success,
        message=message,
        systems=systems,
        candidates=candidates,
        executionTime=round(execution_time, 2),
        lastUpdated=datetime.now().isoformat(),
    )


@app.get("/api/signals/progress")
async def get_progress():
    """Get current progress from the JSONL file."""
    events = []
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    return {
        "events": events[-50:],  # Last 50 events
        "total": len(events),
        "lastUpdated": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
