import React from 'react';
import { DecisionRecord, ErrorLogRecord } from '../api/client';

interface AuditLogTableProps {
  decisions: DecisionRecord[];
  errors: ErrorLogRecord[];
}

export const AuditLogTable: React.FC<AuditLogTableProps> = ({ decisions, errors }) => {
  return (
    <section className="grid grid-cols-1 gap-4 xl:grid-cols-3">
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg xl:col-span-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Bitácora Trazable de Decisiones (SQLite WAL)
        </h3>
        <div className="mt-3 max-h-80 overflow-y-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 border-b border-slate-800 bg-slate-900 text-slate-400">
              <tr>
                <th className="py-2 pr-2">Fecha Vela</th>
                <th className="py-2 pr-2">Ticker</th>
                <th className="py-2 pr-2">Modo</th>
                <th className="py-2 pr-2">r_hat</th>
                <th className="py-2 pr-2">Qwen (API Ext.)</th>
                <th className="py-2 pr-2">Riesgo / Final</th>
                <th className="py-2">Latencia</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              {decisions.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-4 text-center text-slate-500">
                    Sin decisiones registradas todavía.
                  </td>
                </tr>
              ) : (
                decisions.map((d) => (
                  <tr key={d.id} className="hover:bg-slate-800/30">
                    <td className="py-2 pr-2 font-mono text-[11px]">{d.candle_close_utc}</td>
                    <td className="py-2 pr-2 font-semibold">{d.symbol}</td>
                    <td className="py-2 pr-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-[10px] ${
                          d.source_mode === 'live'
                            ? 'bg-cyan-500/20 text-cyan-300'
                            : 'bg-violet-500/20 text-violet-300'
                        }`}
                      >
                        {d.source_mode}
                      </span>
                    </td>
                    <td className="py-2 pr-2 font-mono">
                      {(d.model_pred * 100).toFixed(3)}%
                    </td>
                    <td className="py-2 pr-2">{d.qwen_json?.action || '--'}</td>
                    <td className="py-2 pr-2">
                      <span
                        className={`font-semibold ${
                          d.risk_status === 'APPROVED' ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {d.final_action} ({d.risk_status})
                      </span>
                    </td>
                    <td className="py-2 font-mono">{d.latency_ms.toFixed(1)} ms</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Registro Forense de Errores y Vetos
        </h3>
        <div className="mt-3 max-h-80 space-y-2 overflow-y-auto">
          {errors.length === 0 ? (
            <p className="py-4 text-center text-xs text-slate-500">
              Sin incidentes registrados en error_logs.
            </p>
          ) : (
            errors.map((err) => (
              <div
                key={err.id}
                className="rounded-lg border border-rose-500/20 bg-slate-950/70 p-2.5 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-rose-400">{err.error_code}</span>
                  <span className="text-[10px] text-slate-500">{err.component}</span>
                </div>
                <p className="mt-1 text-slate-300">{err.message}</p>
                <p className="mt-1 font-mono text-[10px] text-slate-500">{err.created_at}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
};
