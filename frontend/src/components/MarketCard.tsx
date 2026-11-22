import React, { useEffect, useState } from 'react';
import { apiClient, CandlesResponse, DecisionRecord, SystemStatusResponse } from '../api/client';

interface MarketCardProps {
  status: SystemStatusResponse | null;
  latestDecision: DecisionRecord | null;
}

export const MarketCard: React.FC<MarketCardProps> = ({ status, latestDecision }) => {
  const ticker = status?.active_ticker || latestDecision?.symbol || 'BTC-USD';
  const closeUtc = latestDecision?.candle_close_utc || 'Sin ciclo ejecutado aún';
  const pred = latestDecision ? latestDecision.model_pred : null;
  const predPct = pred !== null ? (pred * 100).toFixed(4) : '--';
  const [chartData, setChartData] = useState<CandlesResponse | null>(null);
  const [chartLimit, setChartLimit] = useState<number>(60);

  useEffect(() => {
    let active = true;
    apiClient
      .getCandles(ticker, chartLimit)
      .then((res) => {
        if (active) setChartData(res);
      })
      .catch(() => {
        if (active) setChartData(null);
      });
    return () => {
      active = false;
    };
  }, [ticker, chartLimit]);

  const effectiveLastPrice =
    status?.portfolio.last_price && status.portfolio.last_price > 0
      ? status.portfolio.last_price
      : chartData?.last_close || 0;
  const lastPrice = effectiveLastPrice > 0 ? `$${effectiveLastPrice.toLocaleString()}` : '--';

  // Construcción de puntos SVG responsivos para precio + EMA10 + EMA20
  const candles = chartData?.candles || [];
  const width = 760;
  const height = 190;
  const padX = 10;
  const padY = 14;
  const allVals = candles.flatMap((c) => [c.low, c.high, c.ema10, c.ema20]);
  const minVal = allVals.length ? Math.min(...allVals) : 0;
  const maxVal = allVals.length ? Math.max(...allVals) : 1;
  const span = Math.max(maxVal - minVal, 1e-6);

  const toX = (idx: number) =>
    candles.length > 1
      ? padX + (idx / (candles.length - 1)) * (width - padX * 2)
      : width / 2;
  const toY = (val: number) =>
    height - padY - ((val - minVal) / span) * (height - padY * 2);

  const linePath = (getter: (i: number) => number) =>
    candles.map((_c, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(getter(i)).toFixed(1)}`).join(' ');

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wider text-slate-400">Front Office • Activo Evaluado</p>
          <div className="mt-1 flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white">{ticker}</h2>
            {chartData && (
              <span
                className={`rounded px-2 py-0.5 text-xs font-semibold ${
                  chartData.change_pct_1d >= 0
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-rose-500/20 text-rose-300'
                }`}
              >
                {chartData.change_pct_1d >= 0 ? '+' : ''}
                {chartData.change_pct_1d.toFixed(2)}%
              </span>
            )}
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-400">Último Precio Vela Cerrada</p>
          <p className="text-xl font-semibold text-sky-300">{lastPrice}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <p className="text-xs text-slate-400">Vela Cerrada UTC</p>
          <p className="mt-1 truncate text-xs font-mono text-slate-200">{closeUtc}</p>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <p className="text-xs text-slate-400">Pronóstico LSTM T4 (r_hat)</p>
          <p
            className={`mt-1 text-base font-bold ${
              pred === null
                ? 'text-slate-400'
                : pred > 0
                ? 'text-emerald-400'
                : 'text-rose-400'
            }`}
          >
            {pred !== null ? `${pred > 0 ? '+' : ''}${predPct}%` : '--'}
          </p>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <p className="text-xs text-slate-400">Perfil Activo</p>
          <p className="mt-1 text-sm font-medium text-slate-200">
            {status?.simulation_profile || 'agent_paper'}
          </p>
        </div>
        <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
          <p className="text-xs text-slate-400">Velas Procesadas</p>
          <p className="mt-1 text-base font-semibold text-slate-100">
            {status?.processed_candles_count ?? 0}
          </p>
        </div>
      </div>

      {/* Gráfica SVG Responsiva de Velas + EMA10 + EMA20 */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-3">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-3 text-slate-300">
            <span className="font-semibold text-white">Gráfica {ticker}</span>
            <span className="text-amber-300">● EMA 10</span>
            <span className="text-sky-400">● EMA 20</span>
            {chartData && (
              <span className="text-slate-400">
                Máx: ${chartData.high_period.toLocaleString()} / Mín: ${chartData.low_period.toLocaleString()}
              </span>
            )}
          </div>
          <div className="flex gap-1">
            {[30, 60, 140].map((lim) => (
              <button
                key={lim}
                type="button"
                onClick={() => setChartLimit(lim)}
                className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                  chartLimit === lim
                    ? 'bg-sky-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {lim}D
              </button>
            ))}
          </div>
        </div>

        {candles.length > 0 ? (
          <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full overflow-visible">
            {candles.map((c, i) => {
              const x = toX(i);
              const isUp = c.close >= c.open;
              const color = isUp ? '#10b981' : '#f43f5e';
              const bodyTop = toY(Math.max(c.open, c.close));
              const bodyBot = toY(Math.min(c.open, c.close));
              const bodyH = Math.max(1.5, bodyBot - bodyTop);
              const candleW = Math.max(2, (width - padX * 2) / candles.length * 0.55);
              return (
                <g key={c.time}>
                  <line
                    x1={x}
                    y1={toY(c.high)}
                    x2={x}
                    y2={toY(c.low)}
                    stroke={color}
                    strokeWidth="1"
                  />
                  <rect
                    x={x - candleW / 2}
                    y={bodyTop}
                    width={candleW}
                    height={bodyH}
                    fill={color}
                  />
                </g>
              );
            })}
            <path d={linePath((i) => candles[i].ema10)} fill="none" stroke="#fbbf24" strokeWidth="1.5" />
            <path d={linePath((i) => candles[i].ema20)} fill="none" stroke="#38bdf8" strokeWidth="1.5" />
          </svg>
        ) : (
          <div className="flex h-48 items-center justify-center text-xs text-slate-500">
            Cargando velas de {ticker}...
          </div>
        )}
      </div>
    </div>
  );
};
