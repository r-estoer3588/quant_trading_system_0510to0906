"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  Target,
  Activity,
  DollarSign,
  Calendar,
  ArrowLeft,
} from "lucide-react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import type { BacktestResult, BacktestStats, SystemMetrics, TradeRecord } from "@/lib/backtest-types";

// Mock data - Replace with API call
const generateMockBacktestData = (): BacktestResult => {
  const startDate = "2024-01-01";
  const endDate = "2024-12-13";
  const initialCapital = 100000;

  // Generate equity curve
  const equity: { date: string; equity: number; drawdown: number }[] = [];
  let currentEquity = initialCapital;
  let peak = initialCapital;

  for (let i = 0; i < 250; i++) {
    const date = new Date(2024, 0, 1 + i);
    const dailyReturn = (Math.random() - 0.48) * 0.02; // Slight positive bias
    currentEquity *= (1 + dailyReturn);
    peak = Math.max(peak, currentEquity);
    const drawdown = ((peak - currentEquity) / peak) * 100;

    equity.push({
      date: date.toISOString().split("T")[0],
      equity: Math.round(currentEquity),
      drawdown: -drawdown,
    });
  }

  const finalCapital = equity[equity.length - 1]?.equity ?? initialCapital;
  const totalReturn = ((finalCapital - initialCapital) / initialCapital) * 100;
  const maxDrawdown = Math.min(...equity.map(e => e.drawdown));

  // Generate trades
  const trades: TradeRecord[] = [];
  const symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX", "ORCL"];
  const systems = ["System1", "System2", "System3", "System4", "System5", "System6", "System7"];

  for (let i = 0; i < 50; i++) {
    const entryDate = new Date(2024, Math.floor(Math.random() * 11), Math.floor(Math.random() * 28) + 1);
    const holdingDays = Math.floor(Math.random() * 20) + 1;
    const exitDate = new Date(entryDate.getTime() + holdingDays * 24 * 60 * 60 * 1000);
    const entryPrice = 100 + Math.random() * 400;
    const priceChange = (Math.random() - 0.45) * 0.2;
    const exitPrice = entryPrice * (1 + priceChange);
    const side = Math.random() > 0.3 ? "long" : "short";
    const pnlMultiplier = side === "long" ? 1 : -1;
    const pnl = (exitPrice - entryPrice) * 100 * pnlMultiplier;

    trades.push({
      id: i + 1,
      symbol: symbols[Math.floor(Math.random() * symbols.length)] ?? "AAPL",
      side,
      entryDate: entryDate.toISOString().split("T")[0],
      exitDate: exitDate.toISOString().split("T")[0],
      entryPrice: Math.round(entryPrice * 100) / 100,
      exitPrice: Math.round(exitPrice * 100) / 100,
      pnl: Math.round(pnl),
      pnlPercent: Math.round(priceChange * 100 * 100) / 100,
      holdingDays,
      system: systems[Math.floor(Math.random() * systems.length)] ?? "System1",
    });
  }

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl < 0);

  const stats: BacktestStats = {
    totalReturn: Math.round(totalReturn * 100) / 100,
    sharpeRatio: Math.round((totalReturn / 15 + Math.random() * 0.5) * 100) / 100,
    maxDrawdown: Math.round(maxDrawdown * 100) / 100,
    winRate: Math.round((winningTrades.length / trades.length) * 100),
    profitFactor: Math.round((winningTrades.reduce((s, t) => s + t.pnl, 0) / Math.abs(losingTrades.reduce((s, t) => s + t.pnl, 0))) * 100) / 100,
    totalTrades: trades.length,
    avgWin: Math.round(winningTrades.reduce((s, t) => s + t.pnl, 0) / winningTrades.length),
    avgLoss: Math.round(losingTrades.reduce((s, t) => s + t.pnl, 0) / losingTrades.length),
  };

  const systemMetrics: SystemMetrics[] = systems.map((sys, idx) => ({
    system: sys,
    target: 6200,
    filterPass: 1000 + Math.floor(Math.random() * 500),
    setupPass: 50 + Math.floor(Math.random() * 100),
    tradelist: 10,
    entry: Math.floor(Math.random() * 5),
    exit: Math.floor(Math.random() * 3),
  }));

  return {
    stats,
    equity,
    trades: trades.sort((a, b) => new Date(b.entryDate).getTime() - new Date(a.entryDate).getTime()),
    systemMetrics,
    startDate,
    endDate,
    initialCapital,
    finalCapital,
  };
};

function StatCard({ title, value, icon, trend, trendValue }: {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
}) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold font-mono mt-1">{value}</p>
            {trendValue && (
              <p className={`text-xs mt-1 ${
                trend === "up" ? "text-emerald-500" :
                trend === "down" ? "text-red-500" :
                "text-muted-foreground"
              }`}>
                {trendValue}
              </p>
            )}
          </div>
          <div className={`p-3 rounded-lg ${
            trend === "up" ? "bg-emerald-500/10 text-emerald-500" :
            trend === "down" ? "bg-red-500/10 text-red-500" :
            "bg-primary/10 text-primary"
          }`}>
            {icon}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function BacktestPage() {
  const data = generateMockBacktestData();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background/95">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="p-2 rounded-lg hover:bg-muted transition-colors">
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500 bg-clip-text text-transparent">
                  バックテスト結果
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  {data.startDate} 〜 {data.endDate}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono">
                初期資金: ${data.initialCapital.toLocaleString()}
              </Badge>
              <Badge variant="outline" className={data.stats.totalReturn >= 0 ? "text-emerald-500 border-emerald-500/30" : "text-red-500 border-red-500/30"}>
                最終: ${data.finalCapital.toLocaleString()}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
          <StatCard
            title="トータルリターン"
            value={`${data.stats.totalReturn > 0 ? "+" : ""}${data.stats.totalReturn}%`}
            icon={<TrendingUp className="h-5 w-5" />}
            trend={data.stats.totalReturn >= 0 ? "up" : "down"}
          />
          <StatCard
            title="シャープレシオ"
            value={data.stats.sharpeRatio}
            icon={<Activity className="h-5 w-5" />}
            trend={data.stats.sharpeRatio >= 1 ? "up" : "neutral"}
          />
          <StatCard
            title="最大ドローダウン"
            value={`${data.stats.maxDrawdown}%`}
            icon={<TrendingDown className="h-5 w-5" />}
            trend="down"
          />
          <StatCard
            title="勝率"
            value={`${data.stats.winRate}%`}
            icon={<Target className="h-5 w-5" />}
            trend={data.stats.winRate >= 50 ? "up" : "down"}
          />
          <StatCard
            title="プロフィットファクター"
            value={data.stats.profitFactor}
            icon={<BarChart3 className="h-5 w-5" />}
            trend={data.stats.profitFactor >= 1.5 ? "up" : "neutral"}
          />
          <StatCard
            title="総トレード数"
            value={data.stats.totalTrades}
            icon={<DollarSign className="h-5 w-5" />}
          />
        </div>

        <Separator className="my-6" />

        {/* Charts */}
        <Tabs defaultValue="equity" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="equity">エクイティカーブ</TabsTrigger>
            <TabsTrigger value="drawdown">ドローダウン</TabsTrigger>
            <TabsTrigger value="trades">トレード一覧</TabsTrigger>
            <TabsTrigger value="systems">システム別</TabsTrigger>
          </TabsList>

          <TabsContent value="equity">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">資産推移</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={data.equity}>
                    <defs>
                      <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                      tickFormatter={(val) => val.slice(5)}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                      tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                      formatter={(value: number) => [`$${value.toLocaleString()}`, "資産額"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="equity"
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      fill="url(#equityGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="drawdown">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">ドローダウン推移</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={data.equity}>
                    <defs>
                      <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                      tickFormatter={(val) => val.slice(5)}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                      tickFormatter={(val) => `${val.toFixed(1)}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                      formatter={(value: number) => [`${value.toFixed(2)}%`, "ドローダウン"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      strokeWidth={2}
                      fill="url(#ddGradient)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trades">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">トレード履歴（最新20件）</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">銘柄</th>
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">方向</th>
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">エントリー</th>
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">イグジット</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">損益</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">保有日数</th>
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">システム</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.trades.slice(0, 20).map((trade) => (
                        <tr key={trade.id} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-2 font-mono font-medium">{trade.symbol}</td>
                          <td className="py-3 px-2">
                            <Badge variant={trade.side === "long" ? "default" : "secondary"} className="text-xs">
                              {trade.side === "long" ? "買" : "売"}
                            </Badge>
                          </td>
                          <td className="py-3 px-2 text-muted-foreground">{trade.entryDate}</td>
                          <td className="py-3 px-2 text-muted-foreground">{trade.exitDate}</td>
                          <td className={`py-3 px-2 text-right font-mono ${trade.pnl >= 0 ? "text-emerald-500" : "text-red-500"}`}>
                            {trade.pnl >= 0 ? "+" : ""}${trade.pnl.toLocaleString()}
                          </td>
                          <td className="py-3 px-2 text-right font-mono">{trade.holdingDays}日</td>
                          <td className="py-3 px-2">
                            <Badge variant="outline" className="text-xs font-mono">{trade.system}</Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="systems">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">システム別メトリクス</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-2 font-medium text-muted-foreground">システム</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">Tgt</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">FILpass</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">STUpass</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">TRDlist</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">Entry</th>
                        <th className="text-right py-3 px-2 font-medium text-muted-foreground">Exit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.systemMetrics.map((sm) => (
                        <tr key={sm.system} className="border-b hover:bg-muted/50">
                          <td className="py-3 px-2 font-medium">{sm.system}</td>
                          <td className="py-3 px-2 text-right font-mono">{sm.target.toLocaleString()}</td>
                          <td className="py-3 px-2 text-right font-mono">{sm.filterPass.toLocaleString()}</td>
                          <td className="py-3 px-2 text-right font-mono">{sm.setupPass}</td>
                          <td className="py-3 px-2 text-right font-mono font-bold text-violet-500">{sm.tradelist}</td>
                          <td className="py-3 px-2 text-right font-mono text-emerald-500">{sm.entry}</td>
                          <td className="py-3 px-2 text-right font-mono text-amber-500">{sm.exit}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t mt-12 py-4">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          Backtest Dashboard • Next.js + Shadcn/UI + Recharts
        </div>
      </footer>
    </div>
  );
}
