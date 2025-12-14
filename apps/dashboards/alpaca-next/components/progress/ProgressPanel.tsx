"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Loader2, Clock, XCircle } from "lucide-react";

interface ProgressEvent {
  timestamp: string;
  event_type: string;
  data?: {
    system?: string;
    system_name?: string;
    phase?: string;
    phase_name?: string;
    status?: string;
    processed?: number;
    total?: number;
    percentage?: number;
    [key: string]: unknown;
  };
  level?: string;
}

interface Phase {
  id: string;
  label: string;
  status: "pending" | "running" | "complete" | "error";
  progress?: number;
  detail?: string;
}

const PHASES: { id: string; label: string }[] = [
  { id: "initialization", label: "初期化" },
  { id: "load_symbols", label: "銘柄読み込み" },
  { id: "load_cache", label: "キャッシュ読み込み" },
  { id: "filter", label: "フィルター実行" },
  { id: "setup", label: "セットアップ実行" },
  { id: "signals", label: "シグナル抽出" },
  { id: "allocation", label: "配分計算" },
  { id: "complete", label: "完了" },
];

function parseEventsToPhases(events: ProgressEvent[]): Phase[] {
  const phases: Phase[] = PHASES.map(p => ({
    id: p.id,
    label: p.label,
    status: "pending" as const,
  }));

  let currentPhaseIndex = -1;

  for (const event of events) {
    const type = event.event_type?.toLowerCase() || "";
    const data = event.data || {};

    // Detect phase changes
    if (type.includes("initialization") || type.includes("phase0")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 0);
      phases[0].status = type.includes("complete") ? "complete" : "running";
    } else if (type.includes("load_symbol") || type.includes("universe")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 1);
      phases[1].status = type.includes("complete") ? "complete" : "running";
    } else if (type.includes("cache") || type.includes("phase1")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 2);
      phases[2].status = type.includes("complete") ? "complete" : "running";
    } else if (type.includes("filter") || type.includes("phase2")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 3);
      phases[3].status = type.includes("complete") ? "complete" : "running";
      if (data.processed && data.total) {
        phases[3].progress = Math.round((data.processed / data.total) * 100);
        phases[3].detail = `${data.processed}/${data.total}`;
      }
    } else if (type.includes("setup") || type.includes("phase3")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 4);
      phases[4].status = type.includes("complete") ? "complete" : "running";
    } else if (type.includes("signal") || type.includes("phase4") || type.includes("phase5")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 5);
      phases[5].status = type.includes("complete") ? "complete" : "running";
      if (data.processed && data.total) {
        phases[5].progress = Math.round((data.processed / data.total) * 100);
        phases[5].detail = `${data.processed}/${data.total}`;
      }
    } else if (type.includes("allocation") || type.includes("phase6")) {
      currentPhaseIndex = Math.max(currentPhaseIndex, 6);
      phases[6].status = type.includes("complete") ? "complete" : "running";
    } else if (type.includes("complete") || type.includes("done") || type.includes("finish")) {
      if (currentPhaseIndex >= 0) {
        phases[currentPhaseIndex].status = "complete";
      }
    }

    // Mark all previous phases as complete
    for (let i = 0; i < currentPhaseIndex; i++) {
      if (phases[i].status === "running") {
        phases[i].status = "complete";
      }
    }
  }

  return phases;
}

function PhaseItem({ phase }: { phase: Phase }) {
  const statusConfig = {
    pending: {
      icon: <Clock className="h-4 w-4 text-muted-foreground" />,
      bg: "bg-muted/30",
      text: "text-muted-foreground",
    },
    running: {
      icon: <Loader2 className="h-4 w-4 text-emerald-500 animate-spin" />,
      bg: "bg-emerald-500/10",
      text: "text-emerald-500",
    },
    complete: {
      icon: <CheckCircle2 className="h-4 w-4 text-emerald-500" />,
      bg: "bg-emerald-500/5",
      text: "text-muted-foreground",
    },
    error: {
      icon: <XCircle className="h-4 w-4 text-red-500" />,
      bg: "bg-red-500/10",
      text: "text-red-500",
    },
  };

  const config = statusConfig[phase.status];

  return (
    <div className={`flex items-center gap-3 p-2 rounded-lg ${config.bg} transition-all`}>
      {config.icon}
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium ${config.text}`}>{phase.label}</div>
        {phase.progress !== undefined && phase.status === "running" && (
          <div className="mt-1">
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full transition-all duration-300"
                  style={{ width: `${phase.progress}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground font-mono">{phase.detail}</span>
            </div>
          </div>
        )}
      </div>
      {phase.status === "complete" && (
        <Badge variant="outline" className="text-emerald-500 border-emerald-500/30 text-xs">
          完了
        </Badge>
      )}
    </div>
  );
}

interface ProgressPanelProps {
  isRunning: boolean;
  events: ProgressEvent[];
  elapsedTime?: number;
}

export function ProgressPanel({ isRunning, events, elapsedTime }: ProgressPanelProps) {
  const phases = parseEventsToPhases(events);
  const completedCount = phases.filter(p => p.status === "complete").length;
  const runningPhase = phases.find(p => p.status === "running");

  if (!isRunning && events.length === 0) {
    return null;
  }

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
                {elapsedTime !== undefined && ` • ${Math.floor(elapsedTime)}秒`}
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold font-mono text-emerald-500">
              {Math.round((completedCount / phases.length) * 100)}%
            </div>
            <div className="text-xs text-muted-foreground">全体進捗</div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {phases.map((phase) => (
            <PhaseItem key={phase.id} phase={phase} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
