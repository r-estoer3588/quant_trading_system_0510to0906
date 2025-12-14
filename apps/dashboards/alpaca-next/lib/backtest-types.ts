// Type definitions for backtest results
export interface BacktestStats {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgWin: number;
  avgLoss: number;
}

export interface EquityDataPoint {
  date: string;
  equity: number;
  drawdown: number;
}

export interface TradeRecord {
  id: number;
  symbol: string;
  side: "long" | "short";
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  pnlPercent: number;
  holdingDays: number;
  system: string;
}

export interface SystemMetrics {
  system: string;
  target: number;
  filterPass: number;
  setupPass: number;
  tradelist: number;
  entry: number;
  exit: number;
}

export interface BacktestResult {
  stats: BacktestStats;
  equity: EquityDataPoint[];
  trades: TradeRecord[];
  systemMetrics: SystemMetrics[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
}
