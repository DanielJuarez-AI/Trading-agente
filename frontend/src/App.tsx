import React, { useCallback, useEffect, useState } from 'react';
import {
  apiClient,
  BenchmarkResponse,
  DecisionRecord,
  ErrorLogRecord,
  SystemStatusResponse,
} from './api/client';
import { AgentConstellation } from './components/AgentConstellation';
import { AuditLogTable } from './components/AuditLogTable';
import { ConstellationFlowMap } from './components/ConstellationFlowMap';
import { ControlPanel } from './components/ControlPanel';
import { MarketCard } from './components/MarketCard';
import { PortfolioMetrics } from './components/PortfolioMetrics';

export function App() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [errors, setErrors] = useState<ErrorLogRecord[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null);
  const [authenticated, setAuthenticated] = useState(false);
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const refreshAll = useCallback(async () => {
    try {
      const [sess, st, dec, err] = await Promise.all([
        apiClient.checkSession(),
        apiClient.getStatus(),
        apiClient.getDecisions(40),
        apiClient.getErrors(40),
      ]);
      setAuthenticated(sess.authenticated);
      setStatus(st);
      setDecisions(dec.items);
      setErrors(err.items);
      try {
        const bm = await apiClient.getBenchmark();
        setBenchmark(bm);
      } catch {
        // Benchmark opcional hasta que se genere
      }
    } catch (e) {
      setBanner(e instanceof Error ? e.message : 'Error conectando con el backend FastAPI');
    }
  }, []);

  useEffect(() => {
    void refreshAll();
    const timer = window.setInterval(() => {
      void refreshAll();
    }, 8000);
    return () => window.clearInterval(timer);
  }, [refreshAll]);

  const handleLogin = async (credential: string) => {
    setBusy(true);
    setBanner(null);
    try {
      await apiClient.login(credential);
      setAuthenticated(true);
      await refreshAll();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : 'Error de autenticación');
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    try {
      await apiClient.logout();
      setAuthenticated(false);
    } finally {
      setBusy(false);
    }
  };

  const handleAction = async (
    action: 'start' | 'pause' | 'stop' | 'step' | 'replay' | 'switch_ticker' | 'switch_mode' | 'switch_interval' | 'switch_qwen_mode' | 'benchmark',
    payload?: { symbol?: string; mode?: 'live' | 'replay'; interval?: '1h' | '1d'; qwen_mode?: 'EXTERNAL_API' | 'LOCAL_DEVICE'; n_cycles?: number }
  ) => {
    setBusy(true);
    setBanner(null);
    try {
      const res = await apiClient.control(action, payload);
      if (action === 'benchmark') {
        setBenchmark(res as unknown as BenchmarkResponse);
      }
      await refreshAll();
    } catch (e) {
      setBanner(e instanceof Error ? e.message : 'Error ejecutando acción');
      await refreshAll();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 px-4 py-6 text-slate-100 md:px-8">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white md:text-2xl">
              Quant Multi-Agent Trader • Orquestación Híbrida Institucional
            </h1>
            <p className="text-xs text-slate-400">
              Front Office (LSTM T4 + Laya #1 Local) → Qwen 14B (API Externa Remota) → Middle Office (Laya #2 Local + Risk Gate) → Paper Engine
            </p>
          </div>
          <span className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs font-medium text-sky-300">
            Universo Activo: 30 Tickers
          </span>
        </header>

        {banner && (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-xs text-amber-200">
            {banner}
          </div>
        )}

        <ControlPanel
          status={status}
          authenticated={authenticated}
          busy={busy}
          onLogin={handleLogin}
          onLogout={handleLogout}
          onAction={handleAction}
        />

        <MarketCard status={status} latestDecision={decisions[0] || null} />

        <AgentConstellation status={status} latestDecision={decisions[0] || null} />

        <PortfolioMetrics status={status} benchmark={benchmark} />

        <AuditLogTable decisions={decisions} errors={errors} />

        <ConstellationFlowMap status={status} latestDecision={decisions[0] || null} />
      </div>
    </div>
  );
}

export default App;
