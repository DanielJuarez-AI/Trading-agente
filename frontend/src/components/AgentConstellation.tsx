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
  const lat = status?.stage_latencies_ms;
  const fb1 = status?.laya1_feedback;
  const fb2 = status?.laya2_feedback;

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-lg">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Constelación Multi-Agente Híbrida (System-1 Local / System-2 API Externa)
        </h3>
        {(lat?.total_cycle_ms || latestDecision) && (
          <span className="text-xs font-mono text-slate-400">
            Ciclo IA: {(lat?.total_cycle_ms ?? latestDecision?.latency_ms ?? 0).toFixed(2)} ms
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-4">
        {/* Nodo 1: Laya #1 LOCAL */}
        <div className="rounded-lg border border-emerald-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-emerald-300">1. Laya #1 (Alpha)</span>
            <div className="flex items-center gap-1">
              <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-mono text-emerald-300">
                {(lat?.laya1_ms ?? latestDecision?.laya1_json?.latency_ms ?? 0).toFixed(2)} ms
              </span>
              <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                DRL LOCAL
              </span>
            </div>
          </div>
          <p className="mt-1 text-[11px] font-mono text-slate-400">
            {topology?.laya_alpha.location_label || 'Ollama /v1/systemone'}
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>Régimen: <span className="font-semibold text-white">{latestDecision?.laya1_json?.regime || '--'}</span></p>
            <p>Señal: <span className="font-semibold text-white">{latestDecision?.laya1_json?.signal_action || '--'}</span></p>
            <p>Convicción: <span className="font-semibold text-white">{latestDecision?.laya1_json?.conviction_level || '--'} ({latestDecision?.laya1_json?.conviction_score ?? '--'})</span></p>
            {fb1?.guidance && (
              <p className="mt-1 text-[10px] font-mono text-amber-300/90">
                Feedback Qwen (Δτ={fb1.temperature_delta ?? 0}, bias={fb1.conviction_bias ?? 0}): {fb1.guidance}
              </p>
            )}
          </div>
        </div>

        {/* Nodo 2: Qwen (API Externa 14B <-> Local 6GB VRAM 7B-Q4) */}
        <div className="rounded-lg border border-sky-500/40 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-sky-300">
              {topology?.qwen_orchestrator.is_local_device ? '2. Qwen 7B-Q4 (Local)' : '2. Qwen 14B (System-2)'}
            </span>
            <div className="flex items-center gap-1">
              <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-mono text-sky-300">
                {(lat?.qwen_ms ?? latestDecision?.qwen_json?.latency_ms ?? 0).toFixed(2)} ms
              </span>
              <span
                className={`rounded px-2 py-0.5 text-[10px] font-semibold ${
                  topology?.qwen_orchestrator.is_local_device
                    ? 'bg-violet-500/20 text-violet-300'
                    : 'bg-sky-500/20 text-sky-300'
                }`}
              >
                {topology?.qwen_orchestrator.is_local_device ? 'LOCAL 6GB VRAM' : 'API EXTERNA (NO LOCAL)'}
              </span>
            </div>
          </div>
          <p className="mt-1 truncate text-[11px] font-mono text-slate-400">
            {topology?.qwen_orchestrator.endpoint || 'Remote OpenAI-Compatible API'}
          </p>
          <div className="mt-2 space-y-1 text-xs text-slate-300">
            <p>Propuesta: <span className="font-bold text-white">{latestDecision?.qwen_json?.action || '--'}</span></p>
            <p className="line-clamp-3 text-[11px] text-slate-300">
              {latestDecision?.qwen_json?.justification || 'Sin dictamen aún.'}
            </p>
            {fb1 && (
              <p className="mt-1 text-[10px] font-mono text-amber-300/90">
                Orquestador ➔ Laya #1 & #2 ({((lat?.feedback_laya1_ms ?? 0) + (lat?.feedback_laya2_ms ?? 0)).toFixed(2)} ms)
              </p>
            )}
          </div>
        </div>

        {/* Nodo 3: Laya #2 LOCAL */}
        <div className="rounded-lg border border-emerald-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-emerald-300">3. Laya #2 (Guardia)</span>
            <div className="flex items-center gap-1">
              <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-mono text-fuchsia-300">
                {(lat?.laya2_ms ?? latestDecision?.laya2_json?.latency_ms ?? 0).toFixed(2)} ms
              </span>
              <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                LOCAL
              </span>
            </div>
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
            {fb2?.strictness_level && (
              <p className="text-[10px] font-mono text-fuchsia-300/90">
                Feedback Qwen: rigor {fb2.strictness_level} (coh≥{fb2.min_coherence_threshold ?? 0.75})
              </p>
            )}
          </div>
        </div>

        {/* Nodo 4: Motor Determinista de Riesgo */}
        <div className="rounded-lg border border-amber-500/30 bg-slate-950/80 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-bold text-amber-300">4. Risk Hard Gate</span>
            <div className="flex items-center gap-1">
              <span className="rounded bg-slate-900 px-1.5 py-0.5 text-[10px] font-mono text-amber-300">
                {(lat?.risk_paper_ms ?? latestDecision?.latency_ms ?? 0).toFixed(2)} ms
              </span>
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-300">
                HARD GATE
              </span>
            </div>
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
