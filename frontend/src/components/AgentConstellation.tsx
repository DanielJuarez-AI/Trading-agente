import React from 'react';
import { DecisionRecord, SystemStatusResponse } from '../api/client';

interface AgentConstellationProps {
  status: SystemStatusResponse | null;
  latestDecision: DecisionRecord | null;
}

export const AgentConstellation: React.FC<AgentConstellationProps> = ({
  status,
  latestDecision,
}) => {
  const topology = status?.hybrid_topology;

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Constelación Multi-Agente Híbrida (System-1 Local / System-2 API Externa)
        </h3>
        {latestDecision && (
          <span className="text-xs font-mono text-slate-400">
            Latencia total ciclo: {latestDecision.latency_ms.toFixed(1)} ms
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-4">
        {/* Nodo 1: Laya #1 LOCAL */}
        <div className="rounded-lg border border-emerald-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-emerald-300">1. Laya #1 (Alpha)</span>
            <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
              DISPOSITIVO LOCAL
            </span>
          </div>
          <p className="mt-1 text-[11px] font-mono text-slate-400">
            {topology?.laya_alpha.location_label || 'Ollama /v1/systemone'}
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>Régimen: <span className="font-semibold text-white">{latestDecision?.laya1_json?.regime || '--'}</span></p>
            <p>Señal: <span className="font-semibold text-white">{latestDecision?.laya1_json?.signal_action || '--'}</span></p>
            <p>Convicción: <span className="font-semibold text-white">{latestDecision?.laya1_json?.conviction_level || '--'} ({latestDecision?.laya1_json?.conviction_score ?? '--'})</span></p>
          </div>
        </div>

        {/* Nodo 2: Qwen 14B API EXTERNA */}
        <div className="rounded-lg border border-sky-500/40 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-sky-300">2. Qwen 14B (System-2)</span>
            <span className="rounded bg-sky-500/20 px-2 py-0.5 text-[10px] font-semibold text-sky-300">
              API EXTERNA (NO LOCAL)
            </span>
          </div>
          <p className="mt-1 truncate text-[11px] font-mono text-slate-400">
            {topology?.qwen_orchestrator.endpoint || 'Remote OpenAI-Compatible API'}
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>Propuesta: <span className="font-bold text-white">{latestDecision?.qwen_json?.action || '--'}</span></p>
            <p className="line-clamp-3 text-[11px] text-slate-300">
              {latestDecision?.qwen_json?.justification || 'Sin dictamen aún.'}
            </p>
          </div>
        </div>

        {/* Nodo 3: Laya #2 LOCAL */}
        <div className="rounded-lg border border-emerald-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-emerald-300">3. Laya #2 (Guardia)</span>
            <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
              DISPOSITIVO LOCAL
            </span>
          </div>
          <p className="mt-1 text-[11px] font-mono text-slate-400">
            {topology?.laya_guard.location_label || 'Ollama /v1/systemone'}
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>
              Seguridad:{' '}
              <span className="font-semibold text-white">
                {latestDecision?.laya2_json?.security_status || '--'}
              </span>
            </p>
            <p>
              Coherencia:{' '}
              <span className="font-semibold text-white">
                {latestDecision?.laya2_json?.coherence_status || '--'}
              </span>
            </p>
            <p className="line-clamp-2 text-[11px] text-slate-400">
              {latestDecision?.laya2_json?.reason || '--'}
            </p>
          </div>
        </div>

        {/* Nodo 4: Motor Determinista de Riesgo */}
        <div className="rounded-lg border border-amber-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-amber-300">4. Risk Hard Gate</span>
            <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
              PYTHON DETERMINISTA
            </span>
          </div>
          <p className="mt-1 text-[11px] font-mono text-slate-400">
            Inmutable por LLMs (config/risk_limits.yaml)
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>
              Veredicto:{' '}
              <span
                className={`font-bold ${
                  latestDecision?.risk_status === 'APPROVED'
                    ? 'text-emerald-400'
                    : latestDecision?.risk_status === 'REJECTED'
                    ? 'text-rose-400'
                    : 'text-slate-400'
                }`}
              >
                {latestDecision?.risk_status || '--'} ({latestDecision?.final_action || '--'})
              </span>
            </p>
            <p className="line-clamp-2 text-[11px] text-slate-400">
              {latestDecision?.risk_reason || '--'}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
