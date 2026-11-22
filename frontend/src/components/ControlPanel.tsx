import React, { useState } from 'react';
import { OFFICIAL_PRIMARY_TICKERS, SystemStatusResponse } from '../api/client';

interface ControlPanelProps {
  status: SystemStatusResponse | null;
  authenticated: boolean;
  busy: boolean;
  onLogin: (credential: string) => Promise<void>;
  onLogout: () => Promise<void>;
  onAction: (
    action: 'start' | 'pause' | 'stop' | 'step' | 'replay' | 'switch_ticker' | 'switch_mode' | 'benchmark',
    payload?: { symbol?: string; mode?: 'live' | 'replay'; n_cycles?: number }
  ) => Promise<void>;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  status,
  authenticated,
  busy,
  onLogin,
  onLogout,
  onAction,
}) => {
  const [credential, setCredential] = useState('');
  const tickers = status?.primary_tickers?.length
    ? status.primary_tickers
    : OFFICIAL_PRIMARY_TICKERS;
  const activeTicker = status?.active_ticker || 'BTC-USD';
  const sourceMode = status?.source_mode || 'replay';
  const sysState = status?.system_status || 'RUNNING';

  const statusBadgeColor =
    sysState === 'RUNNING'
      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
      : sysState === 'PAUSED'
      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
      : 'bg-rose-500/20 text-rose-300 border-rose-500/40';

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!credential.trim()) return;
    await onLogin(credential.trim());
    setCredential('');
  };

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${statusBadgeColor}`}>
            Estado: {sysState}
          </span>
          <span
            className={`rounded-full border px-3 py-1 text-xs font-semibold ${
              sourceMode === 'live'
                ? 'border-cyan-500/40 bg-cyan-500/20 text-cyan-300'
                : 'border-violet-500/40 bg-violet-500/20 text-violet-300'
            }`}
          >
            {sourceMode === 'live' ? 'LIVE MARKET (MEXC / Público)' : 'HISTORICAL REPLAY'}
          </span>
          <span className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-xs text-slate-300">
            Paper Only (0 Órdenes Reales)
          </span>
        </div>

        {!authenticated ? (
          <form onSubmit={handleLoginSubmit} className="flex items-center gap-2">
            <input
              type="password"
              value={credential}
              onChange={(e) => setCredential(e.target.value)}
              placeholder="Token de operador (Gateway)"
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm text-slate-100 placeholder-slate-500 focus:border-sky-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy}
              className="rounded-lg bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              Autenticar Sesión
            </button>
          </form>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-emerald-400">Sesión HttpOnly Activa</span>
            <button
              onClick={onLogout}
              disabled={busy}
              className="rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-300 hover:bg-slate-800"
            >
              Cerrar Sesión
            </button>
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-xs font-medium text-slate-400">
            Activo ({tickers.length} Tickers):
            <select
              value={activeTicker}
              disabled={!authenticated || busy}
              onChange={(e) => onAction('switch_ticker', { symbol: e.target.value })}
              className="ml-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm text-slate-100 disabled:opacity-50"
            >
              {tickers.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <div className="flex overflow-hidden rounded-lg border border-slate-700">
            <button
              type="button"
              disabled={!authenticated || busy}
              onClick={() => onAction('switch_mode', { mode: 'replay' })}
              className={`px-3 py-1.5 text-xs font-medium ${
                sourceMode === 'replay'
                  ? 'bg-violet-600 text-white'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200'
              } disabled:opacity-50`}
            >
              Historical Replay
            </button>
            <button
              type="button"
              disabled={!authenticated || busy}
              onClick={() => onAction('switch_mode', { mode: 'live' })}
              className={`px-3 py-1.5 text-xs font-medium ${
                sourceMode === 'live'
                  ? 'bg-cyan-600 text-white'
                  : 'bg-slate-950 text-slate-400 hover:text-slate-200'
              } disabled:opacity-50`}
            >
              Live MEXC
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('start')}
            className="rounded-lg bg-emerald-600 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            Iniciar
          </button>
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('pause')}
            className="rounded-lg bg-amber-600 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-amber-500 disabled:opacity-40"
          >
            Pausar
          </button>
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('stop')}
            className="rounded-lg bg-rose-600 px-3.5 py-1.5 text-sm font-medium text-white hover:bg-rose-500 disabled:opacity-40"
          >
            Detener (Kill-Switch)
          </button>
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('step')}
            className="rounded-lg border border-sky-500/40 bg-sky-500/20 px-3.5 py-1.5 text-sm font-medium text-sky-200 hover:bg-sky-500/30 disabled:opacity-40"
          >
            Ejecutar 1 Ciclo
          </button>
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('replay', { symbol: activeTicker, n_cycles: 25 })}
            className="rounded-lg border border-violet-500/40 bg-violet-500/20 px-3.5 py-1.5 text-sm font-medium text-violet-200 hover:bg-violet-500/30 disabled:opacity-40"
          >
            Replay 25 Ciclos
          </button>
          <button
            type="button"
            disabled={!authenticated || busy}
            onClick={() => onAction('benchmark', { symbol: activeTicker, n_cycles: 25 })}
            className="rounded-lg border border-slate-600 bg-slate-800 px-3.5 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-40"
          >
            Benchmark 3 Carriles
          </button>
        </div>
      </div>
    </section>
  );
};
