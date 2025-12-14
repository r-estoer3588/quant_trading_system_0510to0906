"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  Play,
  RefreshCw,
  Settings,
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Clock,
  CheckCircle2,
  XCircle,
  Wifi,
  WifiOff,
} from "lucide-react";
import Link from "next/link";
import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useSignalProgress } from "@/lib/useSignalProgress";
import { ProgressPanel } from "@/components/progress/ProgressPanel";

const SYSTEMS = [
  { id: 1, name: "System1", type: "long", allocation: 25, color: "#8b5cf6" },
  { id: 2, name: "System2", type: "short", allocation: 40, color: "#ef4444" },
  { id: 3, name: "System3", type: "long", allocation: 25, color: "#06b6d4" },
  { id: 4, name: "System4", type: "long", allocation: 25, color: "#10b981" },
  { id: 5, name: "System5", type: "long", allocation: 25, color: "#f59e0b" },
  { id: 6, name: "System6", type: "short", allocation: 40, color: "#ec4899" },
  { id: 7, name: "System7", type: "short", allocation: 20, color: "#6366f1" },
];

interface SystemState {
  system: string;
  target: number;
  filterPass: number;
  setupPass: number;
  tradelist: number;
  entry: number;
  exit: number;
  status: "idle" | "running" | "complete" | "error";
}

interface Candidate {
  symbol: string;
  system: string;
  side: string;
  score: number;
  price: number;
  action: string;
}

const initialSystemStates = (): SystemState[] => {
  return SYSTEMS.map((sys) => ({
    system: sys.name,
    target: 6200,
    filterPass: 0,
    setupPass: 0,
    tradelist: 0,
    entry: 0,
    exit: 0,
    status: "idle" as const,
  }));
};

function SystemCard({ state, config }: { state: SystemState; config: typeof SYSTEMS[0] }) {
  const statusIcon = {
    idle: <Clock className="h-4 w-4 text-muted-foreground" />,
    running: <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />,
    complete: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
    error: <XCircle className="h-4 w-4 text-red-500" />,
  };

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: config.color }} />
            <CardTitle className="text-base">{state.system}</CardTitle>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={config.type === "long" ? "default" : "secondary"}>
              {config.type === "long" ? "買" : "売"}
            </Badge>
            {statusIcon[state.status]}
          </div>
        </div>
        <CardDescription>配分: {config.allocation}%</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2 text-sm">
          <div className="text-center p-2 bg-muted/50 rounded">
            <p className="text-muted-foreground text-xs">FIL</p>
            <p className="font-mono font-medium">{state.filterPass}</p>
          </div>
          <div className="text-center p-2 bg-muted/50 rounded">
            <p className="text-muted-foreground text-xs">STU</p>
            <p className="font-mono font-medium">{state.setupPass}</p>
          </div>
          <div className="text-center p-2 bg-violet-500/10 rounded">
            <p className="text-violet-500 text-xs">TRD</p>
            <p className="font-mono font-bold text-violet-500">{state.tradelist}</p>
          </div>
        </div>
        <div className="flex justify-between mt-3 text-sm">
          <span className="text-emerald-500">Entry: {state.entry}</span>
          <span className="text-amber-500">Exit: {state.exit}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export default function IntegratedPage() {
  const [systemStates, setSystemStates] = useState<SystemState[]>(initialSystemStates());
  const [isRunning, setIsRunning] = useState(false);
  const [capital, setCapital] = useState(100000);
  const [longShare, setLongShare] = useState(50);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [progressLog, setProgressLog] = useState<string[]>([]);

  const { isConnected, events, connect, disconnect, clearEvents } = useSignalProgress();

  // Process new events
  useEffect(() => {
    if (events.length === 0) return;

    const latestEvent = events[events.length - 1];
    if (!latestEvent) return;

    // Update progress log
    const message = latestEvent.data?.phase_name
      ? `${latestEvent.data.phase_name}: ${latestEvent.data.status}`
      : latestEvent.event_type;
    setProgressLog((prev) => [...prev.slice(-20), message]);

    // Update system states based on events
    if (latestEvent.data?.system_name) {
      setSystemStates((prev) =>
        prev.map((s) =>
          s.system.toLowerCase() === latestEvent.data?.system_name?.toLowerCase()
            ? { ...s, status: "running" }
            : s
        )
      );
    }
  }, [events]);

  const runSignals = async () => {
    setIsRunning(true);
    setProgressLog([]);
    clearEvents();

    // Connect WebSocket for progress updates
    connect();

    // Set all systems to running
    setSystemStates((prev) => prev.map((s) => ({ ...s, status: "running" })));

    try {
      const response = await fetch("http://localhost:8000/api/signals/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          capital,
          longShare,
          symbolLimit: 500,
          testMode: "test_symbols",  // 113 symbols for testing, use "" for full
          skipExternal: true,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSystemStates(data.systems.map((s: SystemState) => ({ ...s })));
        setCandidates(data.candidates);
        setExecutionTime(data.executionTime);
        setLastUpdated(new Date());
      } else {
        // Error response
        setSystemStates((prev) => prev.map((s) => ({ ...s, status: "error" })));
      }
    } catch (error) {
      console.error("API error:", error);
      setSystemStates((prev) => prev.map((s) => ({ ...s, status: "error" })));
    } finally {
      // Disconnect WebSocket after completion
      disconnect();
    }

    setLastUpdated(new Date());
    setIsRunning(false);
  };

  const totalEntry = systemStates.reduce((s, st) => s + st.entry, 0);
  const totalExit = systemStates.reduce((s, st) => s + st.exit, 0);
  const totalTradelist = systemStates.reduce((s, st) => s + st.tradelist, 0);

  const allocationData = SYSTEMS.map((sys) => ({
    name: sys.name.replace("System", "S"),
    allocation: sys.allocation,
    color: sys.color,
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background/95">
      <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="p-2 rounded-lg hover:bg-muted transition-colors">
                <ArrowLeft className="h-5 w-5" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-500 bg-clip-text text-transparent">
                  統合ダッシュボード
                </h1>
                <p className="text-sm text-muted-foreground mt-1">System 1-7 シグナル生成</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 text-xs">
                {isConnected ? (
                  <><Wifi className="h-3 w-3 text-emerald-500" /><span className="text-emerald-500">接続中</span></>
                ) : (
                  <><WifiOff className="h-3 w-3 text-muted-foreground" /><span className="text-muted-foreground">未接続</span></>
                )}
              </div>
              <Link href="/backtest" className="px-4 py-2 rounded-lg bg-violet-500/10 hover:bg-violet-500/20 text-violet-500 transition-all">
                バックテスト
              </Link>
              <Button onClick={runSignals} disabled={isRunning}>
                {isRunning ? <><RefreshCw className="h-4 w-4 mr-2 animate-spin" />実行中...</> : <><Play className="h-4 w-4 mr-2" />シグナル生成</>}
              </Button>
            </div>
          </div>
          {lastUpdated && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />最終更新: {lastUpdated.toLocaleString("ja-JP")}
              {executionTime && <span className="ml-2">（実行時間: {executionTime.toFixed(1)}秒）</span>}
            </div>
          )}
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {/* Progress Panel */}
        <ProgressPanel isRunning={isRunning} events={events} />

        <Card className="mb-6">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-lg">設定</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="text-sm text-muted-foreground">資金 (USD)</label>
                <input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))}
                  className="w-full mt-1 px-3 py-2 rounded-lg border bg-background text-foreground font-mono" min={1000} step={1000} />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">ロング配分: {longShare}%</label>
                <input type="range" value={longShare} onChange={(e) => setLongShare(Number(e.target.value))} className="w-full mt-3" min={0} max={100} step={5} />
              </div>
              <div>
                <label className="text-sm text-muted-foreground">ショート配分: {100 - longShare}%</label>
                <div className="flex gap-2 mt-2">
                  <Badge variant="outline" className="text-emerald-500">Long: ${(capital * longShare / 100).toLocaleString()}</Badge>
                  <Badge variant="outline" className="text-red-500">Short: ${(capital * (100 - longShare) / 100).toLocaleString()}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Card><CardContent className="pt-6"><div className="flex items-center justify-between"><div><p className="text-sm text-muted-foreground">TRDlist合計</p><p className="text-2xl font-bold font-mono text-violet-500">{totalTradelist}</p></div><Target className="h-8 w-8 text-violet-500/20" /></div></CardContent></Card>
          <Card><CardContent className="pt-6"><div className="flex items-center justify-between"><div><p className="text-sm text-muted-foreground">Entry候補</p><p className="text-2xl font-bold font-mono text-emerald-500">{totalEntry}</p></div><TrendingUp className="h-8 w-8 text-emerald-500/20" /></div></CardContent></Card>
          <Card><CardContent className="pt-6"><div className="flex items-center justify-between"><div><p className="text-sm text-muted-foreground">Exit候補</p><p className="text-2xl font-bold font-mono text-amber-500">{totalExit}</p></div><TrendingDown className="h-8 w-8 text-amber-500/20" /></div></CardContent></Card>
          <Card><CardContent className="pt-6"><div className="flex items-center justify-between"><div><p className="text-sm text-muted-foreground">システム数</p><p className="text-2xl font-bold font-mono">7</p></div><Activity className="h-8 w-8 text-primary/20" /></div></CardContent></Card>
        </div>

        <Separator className="my-6" />

        <Tabs defaultValue="systems" className="w-full">
          <TabsList className="mb-6">
            <TabsTrigger value="systems">システム状態</TabsTrigger>
            <TabsTrigger value="candidates">候補一覧 ({candidates.length})</TabsTrigger>
            <TabsTrigger value="allocation">配分設定</TabsTrigger>
          </TabsList>

          <TabsContent value="systems">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {systemStates.map((state, idx) => (
                <SystemCard key={state.system} state={state} config={SYSTEMS[idx]!} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="candidates">
            <Card>
              <CardHeader><CardTitle className="text-lg">シグナル候補</CardTitle></CardHeader>
              <CardContent>
                {candidates.length === 0 ? (
                  <p className="text-muted-foreground text-center py-8">シグナル生成を実行してください</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2 px-2">Symbol</th>
                          <th className="text-left py-2 px-2">System</th>
                          <th className="text-left py-2 px-2">Side</th>
                          <th className="text-right py-2 px-2">Score</th>
                          <th className="text-right py-2 px-2">Price</th>
                          <th className="text-left py-2 px-2">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {candidates.slice(0, 20).map((c, i) => (
                          <tr key={i} className="border-b hover:bg-muted/50">
                            <td className="py-2 px-2 font-mono font-bold">{c.symbol}</td>
                            <td className="py-2 px-2">{c.system}</td>
                            <td className="py-2 px-2">
                              <Badge variant={c.side === "long" ? "default" : "secondary"}>
                                {c.side === "long" ? "買" : "売"}
                              </Badge>
                            </td>
                            <td className="py-2 px-2 text-right font-mono">{c.score.toFixed(1)}</td>
                            <td className="py-2 px-2 text-right font-mono">${c.price.toFixed(2)}</td>
                            <td className="py-2 px-2">
                              <Badge variant={c.action === "entry" ? "outline" : "secondary"}>
                                {c.action}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="allocation">
            <Card>
              <CardHeader><CardTitle className="text-lg">システム別配分</CardTitle></CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={allocationData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }} />
                    <YAxis tick={{ fontSize: 12, fill: "hsl(var(--muted-foreground))" }} />
                    <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px" }} />
                    <Bar dataKey="allocation" radius={[4, 4, 0, 0]}>
                      {allocationData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} />))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap gap-2 mt-4 justify-center">
                  {SYSTEMS.map((sys) => (<Badge key={sys.id} variant="outline" style={{ borderColor: sys.color, color: sys.color }}>{sys.name}: {sys.allocation}% ({sys.type})</Badge>))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="border-t mt-12 py-4">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">Integrated Dashboard • Next.js + Shadcn/UI • WebSocket Enabled</div>
      </footer>
    </div>
  );
}
