"use client";

import { MetricCard } from "@/components/metrics/MetricCard";
import { PositionTable } from "@/components/positions/PositionTable";
import { AllocationChart } from "@/components/charts/AllocationChart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { DashboardData, Position, SystemAllocation } from "@/lib/types";
import {
  DollarSign,
  Wallet,
  TrendingUp,
  Activity,
  RefreshCw,
  Clock,
  BarChart3,
} from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";

// Mock data - will be replaced with API calls
const generateMockData = (): DashboardData => ({
  account: {
    equity: 125430.50,
    cash: 45230.00,
    buyingPower: 89500.00,
    lastEquity: 122850.00,
    tradingBlocked: false,
    patternDayTrader: false,
  },
  positions: [
    { symbol: "AAPL", qty: 50, avgEntryPrice: 178.50, currentPrice: 185.20, unrealizedPL: 335.00, unrealizedPLPercent: 3.75, holdingDays: 12, system: "system1" },
    { symbol: "MSFT", qty: 30, avgEntryPrice: 375.80, currentPrice: 390.45, unrealizedPL: 439.50, unrealizedPLPercent: 3.90, holdingDays: 8, system: "system2" },
    { symbol: "NVDA", qty: 20, avgEntryPrice: 485.00, currentPrice: 510.30, unrealizedPL: 506.00, unrealizedPLPercent: 5.22, holdingDays: 15, system: "system1" },
    { symbol: "GOOGL", qty: 25, avgEntryPrice: 142.50, currentPrice: 138.20, unrealizedPL: -107.50, unrealizedPLPercent: -3.02, holdingDays: 22, system: "system3" },
    { symbol: "AMZN", qty: 40, avgEntryPrice: 178.30, currentPrice: 182.45, unrealizedPL: 166.00, unrealizedPLPercent: 2.33, holdingDays: 5, system: "system2" },
    { symbol: "META", qty: 15, avgEntryPrice: 505.20, currentPrice: 485.60, unrealizedPL: -294.00, unrealizedPLPercent: -3.88, holdingDays: 35, system: "system4" },
    { symbol: "TSLA", qty: 35, avgEntryPrice: 242.80, currentPrice: 251.30, unrealizedPL: 297.50, unrealizedPLPercent: 3.50, holdingDays: 10, system: "system3" },
  ],
  allocations: [
    { system: "system1", totalValue: 19379.00, positions: [{ symbol: "AAPL", value: 9260.00 }, { symbol: "NVDA", value: 10206.00 }] },
    { system: "system2", totalValue: 18979.35, positions: [{ symbol: "MSFT", value: 11713.50 }, { symbol: "AMZN", value: 7298.00 }] },
    { system: "system3", totalValue: 12250.50, positions: [{ symbol: "GOOGL", value: 3455.00 }, { symbol: "TSLA", value: 8795.50 }] },
    { system: "system4", totalValue: 7284.00, positions: [{ symbol: "META", value: 7284.00 }] },
  ],
  lastUpdated: new Date().toISOString(),
});

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [tokyoTime, setTokyoTime] = useState<string>("");
  const [nyTime, setNyTime] = useState<string>("");

  const loadData = async () => {
    setLoading(true);
    try {
      // Try API first
      const response = await fetch("http://localhost:8000/api/dashboard");
      if (response.ok) {
        const apiData = await response.json();
        setData(apiData);
        setLastUpdated(new Date());
        setLoading(false);
        return;
      }
    } catch {
      // API failed, fall back to mock data
      console.log("API unavailable, using mock data");
    }
    // Fallback to mock data
    await new Promise((resolve) => setTimeout(resolve, 300));
    setData(generateMockData());
    setLastUpdated(new Date());
    setLoading(false);
  };

  useEffect(() => {
    loadData();

    // Update time on client only to avoid hydration mismatch
    const updateTime = () => {
      const now = new Date();
      setTokyoTime(now.toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", dateStyle: "medium", timeStyle: "short" }));
      setNyTime(now.toLocaleString("ja-JP", { timeZone: "America/New_York", dateStyle: "medium", timeStyle: "short" }));
    };
    updateTime();
    const interval = setInterval(updateTime, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const delta = data ? data.account.equity - data.account.lastEquity : 0;
  const equityRatio = data ? (data.account.buyingPower / data.account.equity) * 100 : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background/95">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 bg-clip-text text-transparent">
                Alpaca <span className="text-foreground/80">現在状況</span>
              </h1>
              <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                <span>🇯🇵 {tokyoTime}</span>
                <span>🇺🇸 {nyTime}</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Link href="/integrated" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 transition-all">
                <Activity className="h-4 w-4" />
                シグナル
              </Link>
              <Link href="/backtest" className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-500 transition-all">
                <BarChart3 className="h-4 w-4" />
                バックテスト
              </Link>
              <Badge variant="outline" className="text-emerald-500 border-emerald-500/30">
                NYSE: クローズ
              </Badge>
              <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/30">
                正常
              </Badge>
              <button
                onClick={loadData}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/10 hover:bg-primary/20 text-primary transition-all disabled:opacity-50"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                更新
              </button>
            </div>
          </div>
          {lastUpdated && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              最終更新: {lastUpdated.toLocaleTimeString("ja-JP")}
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="総資産"
            value={data?.account.equity ?? 0}
            delta={delta}
            deltaLabel="前日比"
            icon={<DollarSign className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="現金"
            value={data?.account.cash ?? 0}
            icon={<Wallet className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="余力"
            value={data?.account.buyingPower ?? 0}
            icon={<TrendingUp className="h-5 w-5" />}
            loading={loading}
          />
          <MetricCard
            title="余力比率"
            value={loading ? "-" : `${equityRatio.toFixed(1)}%`}
            icon={<Activity className="h-5 w-5" />}
            loading={loading}
          />
        </div>

        <Separator className="my-6" />

        {/* Tabs */}
        <Tabs defaultValue="positions" className="w-full">
          <TabsList className="grid w-full max-w-md grid-cols-3 mb-6">
            <TabsTrigger value="summary" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              サマリー
            </TabsTrigger>
            <TabsTrigger value="positions">ポジション</TabsTrigger>
            <TabsTrigger value="allocation">配分</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">ポジション統計</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-card border">
                    <p className="text-sm text-muted-foreground">保有銘柄数</p>
                    <p className="text-2xl font-bold font-mono">{data?.positions.length ?? 0}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-card border">
                    <p className="text-sm text-muted-foreground">勝ちポジション</p>
                    <p className="text-2xl font-bold font-mono text-emerald-500">
                      {data?.positions.filter(p => p.unrealizedPL > 0).length ?? 0}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-card border">
                    <p className="text-sm text-muted-foreground">負けポジション</p>
                    <p className="text-2xl font-bold font-mono text-red-500">
                      {data?.positions.filter(p => p.unrealizedPL < 0).length ?? 0}
                    </p>
                  </div>
                  <div className="p-4 rounded-lg bg-card border">
                    <p className="text-sm text-muted-foreground">合計含み損益</p>
                    <p className={`text-2xl font-bold font-mono ${
                      (data?.positions.reduce((sum, p) => sum + p.unrealizedPL, 0) ?? 0) >= 0
                        ? "text-emerald-500"
                        : "text-red-500"
                    }`}>
                      ${(data?.positions.reduce((sum, p) => sum + p.unrealizedPL, 0) ?? 0).toFixed(2)}
                    </p>
                  </div>
                </div>
              </div>
              <AllocationChart allocations={data?.allocations ?? []} loading={loading} />
            </div>
          </TabsContent>

          <TabsContent value="positions">
            <PositionTable positions={data?.positions ?? []} loading={loading} />
          </TabsContent>

          <TabsContent value="allocation">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <AllocationChart allocations={data?.allocations ?? []} loading={loading} />
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">システム別詳細</h3>
                {data?.allocations.map((alloc) => (
                  <div key={alloc.system} className="p-4 rounded-lg bg-card border">
                    <div className="flex items-center justify-between mb-2">
                      <Badge variant="secondary" className="font-mono">{alloc.system}</Badge>
                      <span className="font-bold font-mono">
                        ${alloc.totalValue.toLocaleString()}
                      </span>
                    </div>
                    <div className="space-y-1">
                      {alloc.positions.map((pos) => (
                        <div key={pos.symbol} className="flex justify-between text-sm text-muted-foreground">
                          <span className="font-mono">{pos.symbol}</span>
                          <span className="font-mono">${pos.value.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t mt-12 py-4">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          Alpaca Dashboard v2.0 • Next.js + Shadcn/UI
        </div>
      </footer>
    </div>
  );
}
