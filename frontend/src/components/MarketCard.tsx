import React, { useEffect, useState } from 'react';
import { apiClient, CandlesResponse, DecisionRecord, SystemStatusResponse } from '../api/client';

interface MarketCardProps {
  status: SystemStatusResponse | null;
  latestDecision: DecisionRecord | null;
}

export const MarketCard: React.FC<MarketCardProps> = ({ status, latestDecision }) => {
  const ticker = status?.active_ticker || latestDecision?.symbol || 'BTC-USD';
  const interval = status?.candle_interval || '1d';
  const closeUtc = latestDecision?.candle_close_utc || 'Sin ciclo ejecutado aún';
  const [chartData, setChartData] = useState<CandlesResponse | null>(null);
  const [chartRangeKey, setChartRangeKey] = useState<string>(interval === '1h' ? '72H' : '60D');

  const rangeOptions =
    interval === '1h'
      ? [
          { key: '24H', label: '1D (24H)', limit: 24, fetchInterval: '1h' as const },
          { key: '72H', label: '3D (72H)', limit: 72, fetchInterval: '1h' as const },
          { key: '168H', label: '7D (168H)', limit: 168, fetchInterval: '1h' as const },
          { key: '240H', label: '10D (240H)', limit: 240, fetchInterval: '1h' as const },
        ]
      : [
          { key: '1D', label: '1D', limit: 24, fetchInterval: '1h' as const },
          { key: '30D', label: '30D', limit: 30, fetchInterval: '1d' as const },
          { key: '60D', label: '60D', limit: 60, fetchInterval: '1d' as const },
          { key: '140D', label: '140D', limit: 140, fetchInterval: '1d' as const },
        ];

  const activeRange =
    rangeOptions.find((opt) => opt.key === chartRangeKey) || rangeOptions[2];

  useEffect(() => {
    setChartRangeKey(interval === '1h' ? '72H' : '60D');
  }, [interval]);

  useEffect(() => {
    let active = true;
    apiClient
      .getCandles(ticker, activeRange.limit, activeRange.fetchInterval)
      .then((res) => {
        if (active) setChartData(res);
      })
      .catch(() => {
        if (active) setChartData(null);
      });
    return () => {
      active = false;
    };
  }, [ticker, activeRange.limit, activeRange.fetchInterval]);

  const af = chartData?.agent_forecast || null;
  const pred = af ? af.pred_return_t1 : latestDecision ? latestDecision.model_pred : null;
  const predPct = pred !== null ? (pred * 100).toFixed(4) : '--';

  const effectiveLastPrice =
    status?.portfolio.last_price && status.portfolio.last_price > 0
      ? status.portfolio.last_price
      : chartData?.last_close || 0;
  const lastPrice = effectiveLastPrice > 0 ? `$${effectiveLastPrice.toLocaleString()}` : '--';

  const candles = chartData?.candles || [];
  const width = 760;
  const height = 210;
  const padX = 12;
  const padRight = af ? 70 : 12;
  const padY = 16;
  const allVals = [
    ...candles.flatMap((c) => [c.low, c.high, c.ema10, c.ema20]),
    ...(af ? [af.projected_close_t1, af.band_upper_t1, af.band_lower_t1] : []),
  ];
  const minVal = allVals.length ? Math.min(...allVals) : 0;
  const maxVal = allVals.length ? Math.max(...allVals) : 1;
  const span = Math.max(maxVal - minVal, 1e-6);

  const toX = (idx: number) =>
    candles.length > 1
      ? padX + (idx / (candles.length - 1)) * (width - padX - padRight)
      : (width - padRight) / 2;
  const toY = (val: number) =>
    height - padY - ((val - minVal) / span) * (height - padY * 2);

  const linePath = (getter: (i: number) => number) =>
    candles
      .map((_c, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(getter(i)).toFixed(1)}`)
      .join(' ');

  const forecastColor =
    af && af.direction === 'UP'
      ? '#10b981'
      : af && af.direction === 'DOWN'
      ? '#f43f5e'
      : '#38bdf8';

  const renderBar = (label: string, pct: number, barClass: string) => {
    const clamped = Math.max(0, Math.min(100, pct));
    return (
      <div key={label} className="space-y-1">
        <div className="flex justify-between text-[11px]">
          <span className="text-slate-300">{label}</span>
          <span className="font-mono text-slate-200">{clamped.toFixed(1)}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barClass}`}
            style={{ width: `${clamped}%` }}
          />
        </div>
      </div>
    );
  };

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

      {/* Gráfica con Pronóstico de Fondo + Panel Lateral de Barras Horizontales de Confianza */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950/80 p-3 lg:col-span-9">
          {af && (
            <div
              className="pointer-events-none absolute inset-0 transition-all duration-500"
              style={{
                background:
                  af.direction === 'UP'
                    ? 'radial-gradient(circle at 80% 45%, rgba(16,185,129,0.12), transparent 65%)'
                    : af.direction === 'DOWN'
                    ? 'radial-gradient(circle at 80% 45%, rgba(244,63,94,0.12), transparent 65%)'
                    : 'radial-gradient(circle at 80% 45%, rgba(56,189,248,0.10), transparent 65%)',
              }}
            />
          )}

          <div className="relative z-10 mb-2 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-3 text-slate-300">
              <span className="font-semibold text-white">Gráfica {ticker} ({interval.toUpperCase()})</span>
              <span className="text-amber-300">● EMA 10</span>
              <span className="text-sky-400">● EMA 20</span>
              {af && <span className="text-emerald-300">┄ Pronóstico {af.horizon_label || (interval === '1h' ? '+1H' : '+1D')}</span>}
            </div>
            <div className="flex gap-1">
              {rangeOptions.map((opt) => (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => setChartRangeKey(opt.key)}
                  className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                    activeRange.key === opt.key
                      ? 'bg-sky-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {candles.length > 0 ? (
            <svg viewBox={`0 0 ${width} ${height}`} className="relative z-10 h-56 w-full overflow-visible">
              {candles.map((c, i) => {
                const x = toX(i);
                const isUp = c.close >= c.open;
                const color = isUp ? '#10b981' : '#f43f5e';
                const bodyTop = toY(Math.max(c.open, c.close));
                const bodyBot = toY(Math.min(c.open, c.close));
                const bodyH = Math.max(1.5, bodyBot - bodyTop);
                const candleW = Math.max(2, ((width - padX - padRight) / candles.length) * 0.55);
                return (
                  <g key={c.time}>
                    <line x1={x} y1={toY(c.high)} x2={x} y2={toY(c.low)} stroke={color} strokeWidth="1" />
                    <rect x={x - candleW / 2} y={bodyTop} width={candleW} height={bodyH} fill={color} />
                  </g>
                );
              })}
              <path d={linePath((i) => candles[i].ema10)} fill="none" stroke="#fbbf24" strokeWidth="1.5" />
              <path d={linePath((i) => candles[i].ema20)} fill="none" stroke="#38bdf8" strokeWidth="1.5" />
              {af && candles.length > 0 && (() => {
                const fc = af.forecast_candle || {
                  open: candles[candles.length - 1].close,
                  high: Math.max(candles[candles.length - 1].close, af.projected_close_t1, af.band_upper_t1),
                  low: Math.min(candles[candles.length - 1].close, af.projected_close_t1, af.band_lower_t1),
                  close: af.projected_close_t1,
                };
                const fcX = width - 28;
                const fcW = Math.max(4, ((width - padX - padRight) / candles.length) * 0.65);
                const fcColor = fc.close >= fc.open ? '#10b981' : '#f43f5e';
                const bodyTop = toY(Math.max(fc.open, fc.close));
                const bodyBot = toY(Math.min(fc.open, fc.close));
                const bodyH = Math.max(3, bodyBot - bodyTop);
                return (
                  <g>
                    <line
                      x1={fcX}
                      y1={toY(fc.high)}
                      x2={fcX}
                      y2={toY(fc.low)}
                      stroke={fcColor}
                      strokeOpacity={0.55}
                      strokeWidth="1.5"
                    />
                    <rect
                      x={fcX - fcW / 2}
                      y={bodyTop}
                      width={fcW}
                      height={bodyH}
                      fill={fcColor}
                      fillOpacity={0.32}
                      stroke={fcColor}
                      strokeOpacity={0.68}
                      strokeWidth="1.2"
                    />
                  </g>
                );
              })()}
            </svg>
          ) : (
            <div className="flex h-56 items-center justify-center text-xs text-slate-500">
              Cargando velas de {ticker}...
            </div>
          )}
        </div>

        {/* Panel Lateral de Barras Horizontales de Confianza */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-3 space-y-3 lg:col-span-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Pronóstico y Confianza
          </h4>
          {af ? (
            <>
              <div className="rounded border border-slate-800 bg-slate-900/80 p-2.5 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-white">{af.signal_action}</span>
                  <span className={af.pred_return_pct >= 0 ? 'font-mono text-emerald-400' : 'font-mono text-rose-400'}>
                    {af.pred_return_pct >= 0 ? '+' : ''}
                    {af.pred_return_pct.toFixed(3)}%
                  </span>
                </div>
                <p className="mt-1 text-[11px] text-slate-400">
                  Obj t+1: <span className="font-mono text-sky-300">${af.projected_close_t1.toLocaleString()}</span>
                </p>
              </div>
              <div className="space-y-2">
                {renderBar(`Convicción (${af.conviction_level})`, af.conviction_score * 100, 'bg-amber-400')}
                {renderBar('Coherencia Laya #2', af.coherence_prob * 100, 'bg-emerald-400')}
                {renderBar('Fuerza / Volatilidad', af.signal_strength * 100, 'bg-sky-400')}
              </div>
              <div className="border-t border-slate-800 pt-2 space-y-1.5">
                <p className="text-[10px] uppercase tracking-wider text-slate-400">Probabilidad Acción (Laya #1)</p>
                {renderBar('BUY', (af.signal_probs.BUY || 0) * 100, 'bg-emerald-500')}
                {renderBar('HOLD', (af.signal_probs.HOLD || 0) * 100, 'bg-sky-500')}
                {renderBar('SELL', (af.signal_probs.SELL || 0) * 100, 'bg-rose-500')}
              </div>
              <div className="border-t border-slate-800 pt-2 space-y-1.5">
                <p className="text-[10px] uppercase tracking-wider text-slate-400">Régimen de Mercado</p>
                {renderBar('Bull Trend', (af.regime_probs.bull_trend || 0) * 100, 'bg-emerald-500')}
                {renderBar('Sideways', (af.regime_probs.sideways || 0) * 100, 'bg-violet-500')}
                {renderBar('Bear Trend', (af.regime_probs.bear_trend || 0) * 100, 'bg-rose-500')}
                {renderBar('High Vol Chop', (af.regime_probs.high_vol_chop || 0) * 100, 'bg-amber-500')}
              </div>
            </>
          ) : (
            <p className="text-xs text-slate-500">Calculando pronóstico...</p>
          )}
        </div>
      </div>
    </div>
  );
};
