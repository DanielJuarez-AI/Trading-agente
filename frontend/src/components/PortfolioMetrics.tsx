import React from 'react';
import { BenchmarkResponse, SystemStatusResponse } from '../api/client';

interface PortfolioMetricsProps {
  status: SystemStatusResponse | null;
  benchmark: BenchmarkResponse | null;
}

export const PortfolioMetrics: React.FC<PortfolioMetricsProps> = ({
  status,
  benchmark,
}) => {
  const p = status?.portfolio;
  const m = status?.metrics;

  const formatWinRate = (val: number | string | undefined) => {
    if (val === undefined) return '--';
    if (typeof val === 'string') return val;
    return `${(val * 100).toFixed(2)}%`;
  };

  return (
    <section className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Back Office • Estado Patrimonial Paper Trading
        </h3>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-6">
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Efectivo</p>
            <p className="mt-1 text-base font-bold text-white">
              ${p ? p.cash.toLocaleString() : '--'}
            </p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Posición Abierta</p>
            <p className="mt-1 text-base font-bold text-sky-300">
              {p ? p.position_qty.toFixed(4) : '--'}
            </p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Patrimonio Total</p>
            <p className="mt-1 text-base font-bold text-emerald-400">
              ${p ? p.total_equity.toLocaleString() : '--'}
            </p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Drawdown Actual</p>
            <p className="mt-1 text-base font-bold text-amber-300">
              {p ? `${p.drawdown_pct.toFixed(2)}%` : '--'}
            </p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Win Rate (Cerradas)</p>
            <p className="mt-1 text-base font-bold text-slate-100">
              {formatWinRate(m?.win_rate)}
            </p>
          </div>
          <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-xs text-slate-400">Órdenes / Cerradas</p>
            <p className="mt-1 text-base font-bold text-slate-100">
              {m ? `${m.total_orders} / ${m.closed_trades}` : '--'}
            </p>
          </div>
        </div>
      </div>

      {benchmark && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              Comparativa Científica 3 Carriles ({benchmark.symbol} • {benchmark.cycles_evaluated} ciclos)
            </h3>
            <span className="text-xs text-slate-400">Perfil: {benchmark.profile}</span>
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-3">Estrategia</th>
                  <th className="py-2 pr-3">Patrimonio Final</th>
                  <th className="py-2 pr-3">Retorno Neto</th>
                  <th className="py-2 pr-3">Max Drawdown</th>
                  <th className="py-2 pr-3">Win Rate</th>
                  <th className="py-2 pr-3">Órdenes</th>
                  <th className="py-2 pr-3">Rechazadas</th>
                  <th className="py-2">Latencia Prom.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 text-slate-200">
                {benchmark.lanes.map((lane) => (
                  <tr key={lane.strategy}>
                    <td className="py-2 pr-3 font-semibold">{lane.strategy}</td>
                    <td className="py-2 pr-3">${lane.final_equity.toLocaleString()}</td>
                    <td
                      className={`py-2 pr-3 font-medium ${
                        lane.net_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {lane.net_return_pct >= 0 ? '+' : ''}
                      {lane.net_return_pct.toFixed(2)}%
                    </td>
                    <td className="py-2 pr-3 text-amber-300">{lane.max_drawdown_pct.toFixed(2)}%</td>
                    <td className="py-2 pr-3">
                      {typeof lane.win_rate === 'number' ? `${lane.win_rate}%` : lane.win_rate}
                    </td>
                    <td className="py-2 pr-3">{lane.total_orders}</td>
                    <td className="py-2 pr-3">{lane.rejected_actions}</td>
                    <td className="py-2 font-mono">{lane.avg_latency_ms.toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 rounded-lg border border-slate-800 bg-slate-950/70 p-2.5 text-xs text-slate-300">
            {benchmark.interpretation}
          </p>
        </div>
      )}
    </section>
  );
};
