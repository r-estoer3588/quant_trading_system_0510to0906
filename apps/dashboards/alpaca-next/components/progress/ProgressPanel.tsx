"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2, Clock, ChevronDown, ChevronRight, Save } from "lucide-react";
import { useState, useEffect, useMemo } from "react";

interface ProgressEvent {
  timestamp: string;
  event_type: string;
  data?: {
    system?: string;
    phase?: string;
    progress?: number;
    processed?: number;
    total?: number;
    candidates?: number;
    candidate_count?: number;
    filter_count?: number;
    setup_count?: number;
    entry_count?: number;
    message?: string;
    [key: string]: unknown;
  };
  level?: string;
}

interface SystemMetrics {
  name: string;
  filterCount: number;
  setupCount: number;
  candidateCount: number;
  entryCount: number;
  progress: number;
  phase: string;
}

interface PhaseInfo {
  id: string;
  label: string;
  status: "pending" | "running" | "complete";
  detail?: string;
  logs: string[];
  systems?: SystemMetrics[];
}

const STORAGE_KEY = "signal_generation_result";

interface SavedResult {
  timestamp: string;
  executionTime: number;
  totalCandidates: number;
  systemSummary: { [key: string]: number };
}

function parseEventsToPhases(events: ProgressEvent[]): {
  phases: PhaseInfo[];
  systemMetrics: { [key: string]: SystemMetrics };
  overallProgress: number;
  dataLoadProgress: { current: number; total: number };
} {
  // Initialize phases
  const phases: PhaseInfo[] = [
    { id: "init", label: "初期化", status: "pending", logs: [] },
    { id: "load_data", label: "銘柄読み込み", status: "pending", logs: [] },
    { id: "cache_check", label: "キャッシュチェック", status: "pending", logs: [] },
    { id: "filter", label: "フィルター実行", status: "pending", logs: [], systems: [] },
    { id: "setup", label: "セットアップ実行", status: "pending", logs: [], systems: [] },
    { id: "signals", label: "シグナル抽出", status: "pending", logs: [], systems: [] },
    { id: "entry", label: "エントリー選定", status: "pending", logs: [], systems: [] },
    { id: "complete", label: "完了", status: "pending", logs: [] },
  ];

  // Initialize system metrics
  const systemMetrics: { [key: string]: SystemMetrics } = {};
  for (let i = 1; i <= 7; i++) {
    systemMetrics[`system${i}`] = {
      name: `System${i}`,
      filterCount: 0,
      setupCount: 0,
      candidateCount: 0,
      entryCount: 0,
      progress: 0,
      phase: "待機中",
    };
  }

  let dataLoadProgress = { current: 0, total: 0 };
  let currentPhase = 0;  // Track current phase index
  let foundSessionStart = false;

  // Find the last session_start to only process events from current session
  let sessionStartIndex = 0;
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].event_type === "session_start") {
      sessionStartIndex = i;
      break;
    }
  }

  // Only process events from current session
  const relevantEvents = events.slice(sessionStartIndex);

  for (const event of relevantEvents) {
    const type = event.event_type?.toLowerCase() || "";
    const data = event.data || {};
    const message = String(data.message || "");
    const timestamp = event.timestamp
      ? new Date(event.timestamp).toLocaleTimeString("ja-JP")
      : "";

    // Handle session_start
    if (type === "session_start") {
      foundSessionStart = true;
      phases[0].status = "running";
      currentPhase = 0;
      continue;
    }

    // Handle stage_update events (most accurate source of system metrics)
    if (type === "stage_update") {
      const sysKey = String(data.system || "").toLowerCase();
      if (sysKey && systemMetrics[sysKey]) {
        const progress = Number(data.progress) || 0;
        const sys = systemMetrics[sysKey];

        sys.progress = Math.max(sys.progress, progress);
        sys.phase = String(data.phase || sys.phase);

        if (data.filter_count !== undefined) {
          sys.filterCount = Number(data.filter_count);
        }
        if (data.setup_count !== undefined) {
          sys.setupCount = Number(data.setup_count);
        }
        if (data.candidate_count !== undefined) {
          sys.candidateCount = Number(data.candidate_count);
        }
        if (data.entry_count !== undefined) {
          sys.entryCount = Number(data.entry_count);
        }

        // Update phase based on progress percentage
        if (progress >= 25 && currentPhase < 3) {
          phases[0].status = "complete";
          phases[1].status = "complete";
          phases[2].status = "complete";
          phases[3].status = "running";
          currentPhase = 3;
        }
        if (progress >= 50 && currentPhase < 4) {
          phases[3].status = "complete";
          phases[4].status = "running";
          currentPhase = 4;
        }
        if (progress >= 75 && currentPhase < 5) {
          phases[4].status = "complete";
          phases[5].status = "running";
          currentPhase = 5;
        }
        if (progress >= 100 && currentPhase < 6) {
          phases[5].status = "complete";
          phases[6].status = "running";
          currentPhase = 6;
        }
      }
      continue;
    }

    // Detect phases from event_type (phaseX_*) and message content
    // Phase 0: Initialization
    if (type.includes("phase0") || type === "initialization") {
      if (type.includes("start") || type === "initialization") {
        phases[0].status = "running";
        currentPhase = 0;
      }
      if (type.includes("complete")) {
        phases[0].status = "complete";
      }
      if (message) {
        phases[0].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 1: Symbol Universe / Data Loading
    else if (type.includes("phase1") || type.includes("symbol_universe") || type === "data_loading") {
      if (currentPhase < 1) {
        phases[0].status = "complete";
        phases[1].status = "running";
        currentPhase = 1;
      }
      if (type.includes("complete")) {
        phases[1].status = "complete";
      }
      if (message) {
        phases[1].logs.push(`${timestamp} ${message}`);
      }
    }
    // Data loading progress (from message parsing)
    else if (message.includes("基礎データロード")) {
      if (currentPhase < 1) {
        phases[0].status = "complete";
        phases[1].status = "running";
        currentPhase = 1;
      }

      const loadMatch = message.match(/進捗[:\s]*(\d+)\/(\d+)/);
      if (loadMatch) {
        dataLoadProgress = {
          current: parseInt(loadMatch[1]),
          total: parseInt(loadMatch[2])
        };
        phases[1].detail = `${dataLoadProgress.current}/${dataLoadProgress.total}`;
      }

      if (message.includes("完了")) {
        phases[1].status = "complete";
        currentPhase = 2;
      }
      phases[1].logs.push(`${timestamp} ${message}`);
    }
    // Phase 2: Cache Check
    else if (type.includes("phase2") || type === "cache_check" || message.includes("Phase 0") || message.includes("rolling キャッシュ")) {
      if (currentPhase < 2) {
        phases[1].status = "complete";
        phases[2].status = "running";
        currentPhase = 2;
      }
      if (type.includes("complete") || message.includes("Phase 0 完了") || message.includes("✅ Phase 0")) {
        phases[2].status = "complete";
        currentPhase = 3;
      }
      if (message) {
        phases[2].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 3: Filter
    else if (type === "filter" || (message.includes("フィルター") && !message.includes("フィルタ通過"))) {
      if (currentPhase < 3) {
        phases[2].status = "complete";
        phases[3].status = "running";
        currentPhase = 3;
      }
      if (message) {
        phases[3].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 4: Setup
    else if (type === "setup" || message.includes("セットアップ")) {
      if (currentPhase < 4) {
        phases[3].status = "complete";
        phases[4].status = "running";
        currentPhase = 4;
      }
      if (message) {
        phases[4].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 5: Signal extraction
    else if (type === "signals" || message.includes("シグナル抽出") || message.includes("トレード候補")) {
      if (currentPhase < 5) {
        phases[4].status = "complete";
        phases[5].status = "running";
        currentPhase = 5;
      }
      if (message) {
        phases[5].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 6: Entry (but NOT "エントリー予定日")
    else if ((type === "entry" && !message.includes("予定日")) ||
             (message.includes("エントリー") && !message.includes("予定日") && message.includes("進捗 100%"))) {
      if (currentPhase < 6) {
        phases[5].status = "complete";
        phases[6].status = "running";
        currentPhase = 6;
      }
      if (message) {
        phases[6].logs.push(`${timestamp} ${message}`);
      }
    }
    // Phase 7: Complete
    else if (type === "complete" || message.includes("実行終了") || message.includes("最終候補件数") || message.includes("Signals generation complete")) {
      phases[6].status = "complete";
      phases[7].status = "complete";
      currentPhase = 7;
      if (message) {
        phases[7].logs.push(`${timestamp} ${message}`);
      }
    }
  }

  // Attach system metrics to phases
  const allSystems = Object.values(systemMetrics);
  phases[3].systems = allSystems.filter(s => s.progress >= 25 || s.filterCount > 0);
  phases[4].systems = allSystems.filter(s => s.progress >= 50 || s.setupCount > 0);
  phases[5].systems = allSystems.filter(s => s.progress >= 75 || s.candidateCount > 0);
  phases[6].systems = allSystems.filter(s => s.progress >= 100 || s.entryCount > 0);

  // Calculate overall progress
  const completedPhases = phases.filter(p => p.status === "complete").length;
  let overallProgress = Math.round((completedPhases / phases.length) * 100);

  // Factor in system progress
  const avgSystemProgress = allSystems.reduce((sum, s) => sum + s.progress, 0) / allSystems.length;
  if (avgSystemProgress > 0 && overallProgress < 100) {
    overallProgress = Math.min(100, Math.round((completedPhases / phases.length * 40) + (avgSystemProgress * 0.6)));
  }

  return { phases, systemMetrics, overallProgress, dataLoadProgress };
}

interface PhaseItemProps {
  phase: PhaseInfo;
  isExpanded: boolean;
  onToggle: () => void;
}

function PhaseItem({ phase, isExpanded, onToggle }: PhaseItemProps) {
  const statusConfig = {
    pending: {
      icon: <Clock className="h-4 w-4 text-muted-foreground" />,
      bg: "bg-muted/30",
      text: "text-muted-foreground",
      label: "待機中",
    },
    running: {
      icon: <Loader2 className="h-4 w-4 text-emerald-500 animate-spin" />,
      bg: "bg-emerald-500/10",
      text: "text-emerald-500",
      label: "実行中",
    },
    complete: {
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
      bg: "bg-emerald-500/5",
      text: "text-muted-foreground",
      label: "完了",
    },
  };

  const config = statusConfig[phase.status];
  const hasContent = phase.logs.length > 0 || (phase.systems && phase.systems.length > 0);

  return (
    <div className="space-y-0">
      <div
        className={`flex items-center gap-3 p-2.5 rounded-lg ${config.bg} transition-all cursor-pointer hover:opacity-80`}
        onClick={onToggle}
      >
        <div className="text-muted-foreground">
          {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </div>
        {config.icon}
        <div className="flex-1 min-w-0">
          <div className={`text-sm font-medium ${config.text}`}>
            {phase.label}
            {phase.detail && <span className="ml-2 text-xs text-muted-foreground">({phase.detail})</span>}
          </div>
        </div>
        <Badge
          variant="outline"
          className={`text-xs ${
            phase.status === "complete" ? "text-emerald-500 border-emerald-500/30" :
            phase.status === "running" ? "text-blue-500 border-blue-500/30" :
            "text-muted-foreground border-muted"
          }`}
        >
          {config.label}
        </Badge>
      </div>

      {isExpanded && (
        <div className="ml-8 mt-1 p-3 bg-muted/20 rounded-lg space-y-3 animate-in slide-in-from-top-1 duration-150">
          {/* System metrics table */}
          {phase.systems && phase.systems.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground mb-2">システム別メトリクス:</div>
              <div className="bg-background/50 rounded overflow-hidden text-xs">
                <div className="grid grid-cols-6 gap-1 font-medium text-muted-foreground p-2 border-b border-muted">
                  <span>System</span>
                  <span className="text-right">Filter</span>
                  <span className="text-right">Setup</span>
                  <span className="text-right">候補</span>
                  <span className="text-right">Entry</span>
                  <span className="text-right">進捗</span>
                </div>
                {phase.systems.map((sys) => (
                  <div key={sys.name} className="grid grid-cols-6 gap-1 p-2 hover:bg-muted/30">
                    <span className="font-medium">{sys.name.replace("System", "S")}</span>
                    <span className="text-right font-mono">{sys.filterCount > 0 ? sys.filterCount : "-"}</span>
                    <span className="text-right font-mono">{sys.setupCount > 0 ? sys.setupCount : "-"}</span>
                    <span className="text-right font-mono text-emerald-500">{sys.candidateCount > 0 ? sys.candidateCount : "-"}</span>
                    <span className="text-right font-mono text-blue-500">{sys.entryCount > 0 ? sys.entryCount : "-"}</span>
                    <span className="text-right font-mono">{sys.progress}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Logs */}
          {phase.logs.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground mb-2">ログ ({phase.logs.length}件):</div>
              <div className="bg-background/50 rounded p-2 max-h-40 overflow-y-auto">
                {phase.logs.slice(-15).map((log, idx) => (
                  <div key={idx} className="text-xs font-mono text-muted-foreground py-0.5 break-words">
                    {log.length > 120 ? log.substring(0, 120) + "..." : log}
                  </div>
                ))}
              </div>
            </div>
          )}

          {!hasContent && (
            <div className="text-xs text-muted-foreground italic">
              {phase.status === "pending" ? "このフェーズはまだ実行されていません" : "詳細データなし"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ProgressPanelProps {
  isRunning: boolean;
  events: ProgressEvent[];
  elapsedTime?: number;
  executionTime?: number;
  totalCandidates?: number;
}

export function ProgressPanel({
  isRunning,
  events,
  elapsedTime,
  executionTime,
  totalCandidates,
}: ProgressPanelProps) {
  const [expandedPhases, setExpandedPhases] = useState<Set<string>>(new Set());
  const [savedResult, setSavedResult] = useState<SavedResult | null>(null);
  const [showSaved, setShowSaved] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) setSavedResult(JSON.parse(saved));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (!isRunning && events.length > 0 && executionTime) {
      const { systemMetrics } = parseEventsToPhases(events);
      const systemSummary: { [key: string]: number } = {};

      Object.values(systemMetrics).forEach(sys => {
        if (sys.candidateCount > 0 || sys.entryCount > 0) {
          systemSummary[sys.name] = sys.entryCount || sys.candidateCount;
        }
      });

      const result: SavedResult = {
        timestamp: new Date().toISOString(),
        executionTime,
        totalCandidates: totalCandidates || 0,
        systemSummary,
      };

      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(result));
        setSavedResult(result);
      } catch { /* ignore */ }
    }
  }, [isRunning, events, executionTime, totalCandidates]);

  const { phases, systemMetrics, overallProgress, dataLoadProgress } = useMemo(
    () => parseEventsToPhases(events),
    [events]
  );

  const togglePhase = (phaseId: string) => {
    setExpandedPhases(prev => {
      const next = new Set(prev);
      if (next.has(phaseId)) next.delete(phaseId);
      else next.add(phaseId);
      return next;
    });
  };

  // Show saved result if no current events
  if (!isRunning && events.length === 0) {
    if (!savedResult) return null;

    return (
      <Card className="border-blue-500/30 bg-gradient-to-br from-blue-500/5 to-transparent mb-6">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center">
                <Save className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <CardTitle className="text-lg text-blue-500">前回の実行結果</CardTitle>
                <p className="text-sm text-muted-foreground">
                  {new Date(savedResult.timestamp).toLocaleString("ja-JP")} • {savedResult.executionTime.toFixed(1)}秒
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold font-mono text-blue-500">{savedResult.totalCandidates}</div>
              <div className="text-xs text-muted-foreground">候補数</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <button
            onClick={() => setShowSaved(!showSaved)}
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-2"
          >
            {showSaved ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            詳細を{showSaved ? "隠す" : "表示"}
          </button>
          {showSaved && Object.keys(savedResult.systemSummary).length > 0 && (
            <div className="grid grid-cols-4 gap-2">
              {Object.entries(savedResult.systemSummary).map(([name, count]) => (
                <div key={name} className="flex justify-between items-center p-2 bg-muted/30 rounded text-sm">
                  <span>{name.replace("System", "S")}</span>
                  <span className="font-mono text-emerald-500">{count}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  const completedCount = phases.filter(p => p.status === "complete").length;
  const runningPhase = phases.find(p => p.status === "running");

  return (
    <Card className="border-emerald-500/30 bg-gradient-to-br from-emerald-500/5 to-transparent mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isRunning ? (
              <div className="relative">
                <div className="w-10 h-10 rounded-full border-4 border-emerald-500/30 flex items-center justify-center">
                  <Loader2 className="h-5 w-5 text-emerald-500 animate-spin" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full animate-pulse" />
              </div>
            ) : (
              <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              </div>
            )}
            <div>
              <CardTitle className="text-lg text-emerald-500">
                {isRunning ? "シグナル生成中..." : "完了"}
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                {runningPhase ? runningPhase.label : `${completedCount}/${phases.length} フェーズ完了`}
                {dataLoadProgress.total > 0 && runningPhase?.id === "load_data" && (
                  <span className="ml-2">• {dataLoadProgress.current}/{dataLoadProgress.total}</span>
                )}
                {(elapsedTime || executionTime) && ` • ${Math.floor(elapsedTime || executionTime || 0)}秒`}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold font-mono text-emerald-500">{overallProgress}%</div>
            <div className="text-xs text-muted-foreground">全体進捗</div>
          </div>
        </div>

        {/* Data loading progress bar */}
        {dataLoadProgress.total > 0 && (
          <div className="mt-3">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>銘柄読み込み</span>
              <span>{dataLoadProgress.current}/{dataLoadProgress.total}</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                style={{ width: `${(dataLoadProgress.current / dataLoadProgress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent>
        <p className="text-xs text-muted-foreground mb-2">▶ 各フェーズをクリックして詳細を表示</p>
        <div className="space-y-1">
          {phases.map((phase) => (
            <PhaseItem
              key={phase.id}
              phase={phase}
              isExpanded={expandedPhases.has(phase.id)}
              onToggle={() => togglePhase(phase.id)}
            />
          ))}
        </div>

        {/* System Summary at bottom */}
        {Object.values(systemMetrics).some(s => s.progress > 0 || s.filterCount > 0) && (
          <div className="mt-4 pt-4 border-t border-muted">
            <div className="text-xs font-medium text-muted-foreground mb-2">システム進捗サマリー:</div>
            <div className="grid grid-cols-7 gap-1 text-xs">
              {Object.values(systemMetrics).map(sys => (
                <div key={sys.name} className="text-center p-2 bg-muted/30 rounded">
                  <div className="font-medium">{sys.name.replace("System", "S")}</div>
                  <div className="font-mono text-emerald-500">{sys.progress}%</div>
                  {(sys.entryCount > 0 || sys.candidateCount > 0) && (
                    <div className="text-muted-foreground text-[10px]">
                      {sys.entryCount > 0 ? `E:${sys.entryCount}` : `C:${sys.candidateCount}`}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
