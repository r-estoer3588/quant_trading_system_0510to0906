'use client';

import type { AlpacaPosition, AlpacaSnapshot } from '@/lib/types';
import { fmtPrice } from '@/lib/format';
import { AlpacaSection as LegacyAlpacaSection } from '@/components/AlpacaSectionLegacy';

type ProtectionPosition = AlpacaPosition & {
  stop_price_source?: 'broker_order' | 'broker_hwm' | 'soft_hwm' | null;
  trailing_hwm?: number | null;
  protection_mode?: 'native_trailing_expected' | 'synthetic_intraday' | string | null;
  protection_observed?: 'trailing' | 'stop' | 'none' | 'unmeasured' | string | null;
  protection_state?:
    | 'verified'
    | 'soft_monitored'
    | 'soft_unmeasured'
    | 'pending_arm'
    | 'missing_after_arm'
    | 'unmeasured'
    | string
    | null;
  protection_verified?: boolean;
  protection_observed_at?: string | null;
  protection_order_id?: string | null;
  resting_client_order_id?: string | null;
};

function protectionLabel(p: ProtectionPosition): { text: string; cls: string; sub: string } {
  const pct =
    p.trailing_stop_pct != null && Number.isFinite(p.trailing_stop_pct)
      ? `${(p.trailing_stop_pct * 100).toFixed(0)}%`
      : '?';
  const stop = p.stop_price_est != null ? fmtPrice(p.stop_price_est) : '—';
  const hwm = p.trailing_hwm != null ? ` · HWM ${fmtPrice(p.trailing_hwm)}` : '';

  switch (p.protection_state) {
    case 'verified':
      return {
        text: `trail ${pct} ✓ broker · stop ${stop}`,
        cls: 'border-ok/25 bg-ok/10 text-ok',
        sub: `${p.protection_observed ?? 'broker'}${hwm}`,
      };
    case 'soft_monitored':
      return {
        text: `soft trail ${pct} ⚠ · stop ${stop}`,
        cls: 'border-warn/25 bg-warn/10 text-warn',
        sub: `30分監視 · HWM ratchet${hwm}`,
      };
    case 'soft_unmeasured':
      return {
        text: `soft trail ${pct} ⚠ · threshold未計測`,
        cls: 'border-warn/25 bg-warn/10 text-warn',
        sub: 'legacy fractional · HWM待ち',
      };
    case 'pending_arm':
      return {
        text: `trail ${pct} · pending arm`,
        cls: 'border-sky-400/20 bg-sky-400/10 text-sky-300',
        sub: '本日entry · broker arm待ち',
      };
    case 'missing_after_arm':
      return {
        text: `UNPROTECTED · trail ${pct}`,
        cls: 'border-fail/35 bg-fail/15 text-fail',
        sub: 'OPEN order観測済み・resting protection無し',
      };
    default:
      return {
        text: `trail ${pct} ?`,
        cls: 'border-white/10 bg-white/[0.03] text-muted',
        sub: 'broker observation unmeasured',
      };
  }
}

function ProtectionStrip({ positions }: { positions: ProtectionPosition[] }) {
  const rows = positions.filter(
    (p) => p.protection_state != null && (p.system === 'system1' || p.system === 'system4'),
  );
  if (rows.length === 0) return null;
  const faults = rows.filter((p) => p.protection_state === 'missing_after_arm').length;
  const verified = rows.filter((p) => p.protection_state === 'verified').length;
  const soft = rows.filter((p) => p.protection_state === 'soft_monitored').length;

  return (
    <section className="mb-3 rounded-xl border border-white/10 bg-card p-3 shadow-lg">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <span className="text-xs font-semibold text-cardfg">S1/S4 trailing protection</span>
          <span className="ml-2 text-[10px] text-muted">
            broker verified {verified} · soft {soft} · fault {faults}
          </span>
        </div>
        <span className="text-[9px] text-muted/60">OPEN orders → broker HWM → soft HWM</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {rows.map((p) => {
          const label = protectionLabel(p);
          return (
            <div
              key={`${p.system}-${p.symbol}`}
              className={`rounded-md border px-2 py-1 text-[10px] leading-tight ${label.cls}`}
              title={p.resting_client_order_id ?? p.protection_order_id ?? undefined}
            >
              <span className="font-semibold">{p.symbol}</span>
              <span className="ml-1">{label.text}</span>
              <div className="mt-0.5 text-[9px] opacity-70">{label.sub}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * Authoritative protection wrapper around the existing dashboard section.
 *
 * The strip states trailing intent + broker/soft observation explicitly. For a
 * concrete measured threshold we also adapt the legacy row badge to its generic
 * stop-price renderer so the row shows the same real stop number instead of the
 * old ATR/broker-unverified copy. Accounting and all other rendering stay in the
 * preserved legacy component.
 */
export function AlpacaSection({ payload }: { payload: AlpacaSnapshot | null }) {
  if (!payload) return <LegacyAlpacaSection payload={payload} />;
  const positions = payload.positions as ProtectionPosition[];
  const adapted: AlpacaSnapshot = {
    ...payload,
    positions: positions.map((p) => {
      if (
        p.stop_price_est != null &&
        (p.protection_state === 'verified' || p.protection_state === 'soft_monitored')
      ) {
        return { ...p, exit_type: 'stop' };
      }
      if (p.protection_state === 'missing_after_arm') {
        return { ...p, exit_type: 'unknown' };
      }
      return p;
    }),
  };
  return (
    <>
      <ProtectionStrip positions={positions} />
      <LegacyAlpacaSection payload={adapted} />
    </>
  );
}
