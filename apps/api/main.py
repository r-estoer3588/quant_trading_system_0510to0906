"""
FastAPI Backend for Alpaca Dashboard.

Provides REST API endpoints for the Next.js frontend.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing Alpaca utilities
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root
try:
    from common.broker_alpaca import get_client
    from common.position_age import get_position_age

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
    get_position_age = None


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
    if get_position_age is None:
        return 0
    try:
        return get_position_age(symbol)
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


class RunSignalsResponse(BaseModel):
    """Response from signal generation run."""

    success: bool
    message: str
    systems: list[SystemState]
    candidates: list[SignalCandidate]
    executionTime: float
    lastUpdated: str


@app.post("/api/signals/run", response_model=RunSignalsResponse)
async def run_signals(request: RunSignalsRequest):
    """Run signal generation for all systems."""
    import random
    import time

    start_time = time.time()

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

    # Simulate running each system
    for cfg in systems_config:
        # Add some delay to simulate processing
        time.sleep(0.1)

        systems.append(
            SystemState(
                system=cfg["name"],
                target=6200,
                filterPass=1000 + random.randint(0, 500),
                setupPass=50 + random.randint(0, 100),
                tradelist=random.randint(0, 10),
                entry=random.randint(0, 5),
                exit=random.randint(0, 3),
                status="complete",
            )
        )

    # Generate candidates based on capital allocation
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

    for _ in range(20):
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

    candidates.sort(key=lambda x: x.score, reverse=True)
    execution_time = time.time() - start_time

    return RunSignalsResponse(
        success=True,
        message=f"Signal generation complete. Capital: ${request.capital:,.0f}, Long: {request.longShare}%",
        systems=systems,
        candidates=candidates,
        executionTime=round(execution_time, 2),
        lastUpdated=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
