"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Position } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ArrowUpDown, TrendingUp, TrendingDown } from "lucide-react";
import { useState } from "react";

interface PositionTableProps {
  positions: Position[];
  loading?: boolean;
}

type SortKey = "symbol" | "unrealizedPL" | "unrealizedPLPercent" | "holdingDays";
type SortOrder = "asc" | "desc";

export function PositionTable({ positions, loading = false }: PositionTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("unrealizedPL");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("desc");
    }
  };

  const sortedPositions = [...positions].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (typeof aVal === "string" && typeof bVal === "string") {
      return sortOrder === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    }
    return sortOrder === "asc" ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
  });

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
    }).format(val);

  const formatPercent = (val: number) =>
    `${val >= 0 ? "+" : ""}${val.toFixed(2)}%`;

  if (loading) {
    return (
      <div className="rounded-lg border bg-card">
        <div className="p-4 space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-12 animate-pulse rounded bg-muted" />
          ))}
        </div>
      </div>
    );
  }

  if (positions.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
        ポジションはありません
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead
              className="cursor-pointer hover:text-foreground transition-colors"
              onClick={() => handleSort("symbol")}
            >
              <div className="flex items-center gap-2">
                銘柄
                <ArrowUpDown className="h-4 w-4" />
              </div>
            </TableHead>
            <TableHead className="text-right">数量</TableHead>
            <TableHead className="text-right">平均取得単価</TableHead>
            <TableHead className="text-right">現在値</TableHead>
            <TableHead
              className="text-right cursor-pointer hover:text-foreground transition-colors"
              onClick={() => handleSort("unrealizedPL")}
            >
              <div className="flex items-center justify-end gap-2">
                含み損益
                <ArrowUpDown className="h-4 w-4" />
              </div>
            </TableHead>
            <TableHead
              className="text-right cursor-pointer hover:text-foreground transition-colors"
              onClick={() => handleSort("unrealizedPLPercent")}
            >
              <div className="flex items-center justify-end gap-2">
                損益率
                <ArrowUpDown className="h-4 w-4" />
              </div>
            </TableHead>
            <TableHead
              className="text-right cursor-pointer hover:text-foreground transition-colors"
              onClick={() => handleSort("holdingDays")}
            >
              <div className="flex items-center justify-end gap-2">
                保有日数
                <ArrowUpDown className="h-4 w-4" />
              </div>
            </TableHead>
            <TableHead>システム</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sortedPositions.map((position) => (
            <TableRow
              key={position.symbol}
              className={cn(
                "transition-colors",
                position.unrealizedPL > 0 && "bg-emerald-500/5 hover:bg-emerald-500/10",
                position.unrealizedPL < 0 && "bg-red-500/5 hover:bg-red-500/10"
              )}
            >
              <TableCell className="font-mono font-semibold">
                {position.symbol}
              </TableCell>
              <TableCell className="text-right font-mono">
                {position.qty}
              </TableCell>
              <TableCell className="text-right font-mono">
                {formatCurrency(position.avgEntryPrice)}
              </TableCell>
              <TableCell className="text-right font-mono">
                {formatCurrency(position.currentPrice)}
              </TableCell>
              <TableCell className="text-right">
                <div className="flex items-center justify-end gap-2">
                  {position.unrealizedPL > 0 ? (
                    <TrendingUp className="h-4 w-4 text-emerald-500" />
                  ) : position.unrealizedPL < 0 ? (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  ) : null}
                  <span
                    className={cn(
                      "font-mono font-medium",
                      position.unrealizedPL > 0 && "text-emerald-500",
                      position.unrealizedPL < 0 && "text-red-500"
                    )}
                  >
                    {formatCurrency(position.unrealizedPL)}
                  </span>
                </div>
              </TableCell>
              <TableCell className="text-right">
                <Badge
                  variant="outline"
                  className={cn(
                    "font-mono",
                    position.unrealizedPLPercent > 0 &&
                      "border-emerald-500/50 bg-emerald-500/10 text-emerald-500",
                    position.unrealizedPLPercent < 0 &&
                      "border-red-500/50 bg-red-500/10 text-red-500"
                  )}
                >
                  {formatPercent(position.unrealizedPLPercent)}
                </Badge>
              </TableCell>
              <TableCell className="text-right font-mono">
                <span
                  className={cn(
                    position.holdingDays > 30 && "text-amber-500 font-semibold"
                  )}
                >
                  {position.holdingDays}日
                </span>
              </TableCell>
              <TableCell>
                <Badge variant="secondary" className="font-mono">
                  {position.system}
                </Badge>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
