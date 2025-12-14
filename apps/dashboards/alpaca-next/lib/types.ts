// Type definitions for Alpaca Dashboard
export interface Position {
  symbol: string;
  qty: number;
  avgEntryPrice: number;
  currentPrice: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  holdingDays: number;
  system: string;
  sparklineData?: number[];
}

export interface AccountInfo {
  equity: number;
  cash: number;
  buyingPower: number;
  lastEquity: number;
  tradingBlocked: boolean;
  patternDayTrader: boolean;
}

export interface SystemAllocation {
  system: string;
  totalValue: number;
  positions: { symbol: string; value: number }[];
}

export interface DashboardData {
  account: AccountInfo;
  positions: Position[];
  allocations: SystemAllocation[];
  lastUpdated: string;
}
