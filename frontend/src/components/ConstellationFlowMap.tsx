import React, { useEffect, useRef, useState } from 'react';
import { DecisionRecord, SystemStatusResponse } from '../api/client';

interface ConstellationFlowMapProps {
  status: SystemStatusResponse | null;
  latestDecision: DecisionRecord | null;
}

const STAR_NODES = [
  { id: 'market', rx: 0.08, ry: 0.34, name: 'Mercado Público', tag: 'BACK OFFICE', color: '#38bdf8', r: 9 },
  { id: 'ingest', rx: 0.21, ry: 0.22, name: 'Ingesta & Idempotencia', tag: 'SQLITE WAL', color: '#60a5fa', r: 8 },
  { id: 'lstm', rx: 0.34, ry: 0.42, name: 'LSTM T4 Alpha', tag: 'FRONT OFFICE', color: '#34d399', r: 10 },
  { id: 'laya1', rx: 0.47, ry: 0.23, name: 'Laya #1 (Local)', tag: 'SYSTEM-1 LOCAL', color: '#10b981', r: 10 },
  { id: 'gateway', rx: 0.59, ry: 0.43, name: 'Auth Gateway', tag: 'X-API-KEY', color: '#818cf8', r: 8 },
  { id: 'qwen', rx: 0.71, ry: 0.24, name: 'Qwen 14B (API Ext.)', tag: 'SYSTEM-2', color: '#a78bfa', r: 10 },
  { id: 'laya2', rx: 0.81, ry: 0.48, name: 'Laya #2 (Local)', tag: 'MIDDLE OFFICE', color: '#10b981', r: 10 },
  { id: 'risk', rx: 0.92, ry: 0.31, name: 'Risk Gate & Paper', tag: 'HARD GATE', color: '#fbbf24', r: 11 },
  { id: 'veto', rx: 0.86, ry: 0.79, name: 'Veto / Pausa Segura', tag: 'ERROR_LOGS', color: '#f43f5e', r: 8 },
];

interface StarEdge {
  from: number;
  to: number;
  step: number;
  isVeto?: boolean;
  isFeedback?: boolean;
  curve?: number;
  fbColor?: string;
}

const STAR_EDGES: StarEdge[] = [
  { from: 0, to: 1, step: 0 },
  { from: 1, to: 2, step: 1 },
  { from: 2, to: 3, step: 2 },
  { from: 2, to: 6, step: 4 },
  { from: 3, to: 4, step: 3 },
  { from: 4, to: 5, step: 4 },
  { from: 5, to: 3, step: 4.5, curve: -74, isFeedback: true, fbColor: '#f59e0b' },
  { from: 5, to: 6, step: 4.9, curve: 46, isFeedback: true, fbColor: '#e879f9' },
  { from: 5, to: 6, step: 5.2 },
  { from: 6, to: 7, step: 6 },
  { from: 6, to: 8, step: 6, isVeto: true },
  { from: 7, to: 8, step: 7, isVeto: true },
];

const BG_STARS = Array.from({ length: 85 }, (_, idx) => ({
  x: ((idx * 173 + 31) % 997) / 997,
  y: ((idx * 317 + 71) % 991) / 991,
  size: idx % 3 === 0 ? 1.8 : 1.0,
  phase: idx * 0.65,
}));

export const ConstellationFlowMap: React.FC<ConstellationFlowMapProps> = ({
  status,
  latestDecision,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pulseStartRef = useRef<number>(performance.now());
  const [selectedIdx, setSelectedIdx] = useState<number>(2);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let rafId = 0;
    const resize = () => {
      if (!canvas.parentElement) return;
      canvas.width = Math.max(320, Math.floor(canvas.parentElement.clientWidth));
      canvas.height = 420;
    };
    resize();
    window.addEventListener('resize', resize);

    const draw = (now: number) => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const bgGrad = ctx.createRadialGradient(w * 0.5, h * 0.4, 20, w * 0.5, h * 0.5, w * 0.75);
      bgGrad.addColorStop(0, '#090f2b');
      bgGrad.addColorStop(0.55, '#040819');
      bgGrad.addColorStop(1, '#02040d');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      BG_STARS.forEach((s) => {
        const alpha = 0.22 + 0.35 * (0.5 + 0.5 * Math.sin(now * 0.0018 + s.phase));
        ctx.fillStyle = `rgba(226, 232, 240, ${alpha.toFixed(2)})`;
        ctx.beginPath();
        ctx.arc(s.x * w, s.y * h, s.size, 0, Math.PI * 2);
        ctx.fill();
      });

      const elapsedSec = ((now - pulseStartRef.current) / 1000.0) % 5.6;
      const activeStep = (elapsedSec / 5.6) * 7.5;
      const isVetoed =
        latestDecision?.risk_status === 'REJECTED' || status?.system_status === 'STOPPED';

      const getStarMs = (nodeId: string): number | null => {
        const lat = status?.stage_latencies_ms ?? latestDecision?.qwen_json?.stage_latencies_ms;
        switch (nodeId) {
          case 'ingest':
            return lat?.ingest_ms ?? null;
          case 'lstm':
            return lat?.lstm_ms ?? null;
          case 'laya1':
            return lat?.laya1_ms ?? latestDecision?.laya1_json?.latency_ms ?? null;
          case 'gateway':
            return 0.1;
          case 'qwen':
            return lat?.qwen_ms ?? latestDecision?.qwen_json?.latency_ms ?? null;
          case 'laya2':
            return lat?.laya2_ms ?? latestDecision?.laya2_json?.latency_ms ?? null;
          case 'risk':
            return lat?.risk_paper_ms ?? lat?.total_cycle_ms ?? latestDecision?.latency_ms ?? null;
          default:
            return null;
        }
      };

      const quadPt = (x1: number, y1: number, cx: number, cy: number, x2: number, y2: number, t: number) => {
        const mt = 1 - t;
        return {
          x: mt * mt * x1 + 2 * mt * t * cx + t * t * x2,
          y: mt * mt * y1 + 2 * mt * t * cy + t * t * y2,
        };
      };

      STAR_EDGES.forEach((edge) => {
        const a = STAR_NODES[edge.from];
        const b = STAR_NODES[edge.to];
        const x1 = a.rx * w;
        const y1 = a.ry * h;
        const x2 = b.rx * w;
        const y2 = b.ry * h;
        const hasCurve = typeof edge.curve === 'number';
        const cx = (x1 + x2) * 0.5;
        const cy = (y1 + y2) * 0.5 + (edge.curve || 0);

        ctx.save();
        if (edge.isFeedback) {
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = edge.fbColor === '#e879f9' ? 'rgba(232, 121, 249, 0.55)' : 'rgba(245, 158, 11, 0.55)';
          ctx.lineWidth = 2.0;
        } else if (edge.isVeto) {
          ctx.setLineDash([4, 5]);
          ctx.strokeStyle = isVetoed ? 'rgba(244, 63, 94, 0.55)' : 'rgba(244, 63, 94, 0.16)';
          ctx.lineWidth = 1.6;
        } else {
          ctx.strokeStyle = 'rgba(129, 140, 248, 0.28)';
          ctx.lineWidth = 1.6;
        }
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        if (hasCurve) {
          ctx.quadraticCurveTo(cx, cy, x2, y2);
        } else {
          ctx.lineTo(x2, y2);
        }
        ctx.stroke();

        if (edge.isFeedback) {
          const mid = quadPt(x1, y1, cx, cy, x2, y2, 0.5);
          ctx.fillStyle = edge.fbColor || '#f59e0b';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center';
          ctx.fillText(
            edge.from === 5 && edge.to === 3 ? 'Feedback DRL (Δτ, bias)' : 'Feedback Guardia (rigor)',
            mid.x,
            mid.y - 6
          );
        }
        ctx.restore();

        const shouldPulse = edge.isVeto ? isVetoed : true;
        const localProg = activeStep - edge.step;
        if (shouldPulse && localProg >= 0 && localProg <= 1.0) {
          const p = hasCurve
            ? quadPt(x1, y1, cx, cy, x2, y2, localProg)
            : { x: x1 + (x2 - x1) * localProg, y: y1 + (y2 - y1) * localProg };
          const tTail = Math.max(0, localProg - 0.18);
          const tail = hasCurve
            ? quadPt(x1, y1, cx, cy, x2, y2, tTail)
            : { x: x1 + (x2 - x1) * tTail, y: y1 + (y2 - y1) * tTail };
          const cometColor = edge.isFeedback ? edge.fbColor || '#f59e0b' : edge.isVeto ? '#fb7185' : b.color;
          const grad = ctx.createLinearGradient(tail.x, tail.y, p.x, p.y);
          grad.addColorStop(0, 'rgba(255,255,255,0)');
          grad.addColorStop(1, cometColor);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 3.2;
          ctx.beginPath();
          ctx.moveTo(tail.x, tail.y);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();

          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(p.x, p.y, 3.6, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      STAR_NODES.forEach((n, idx) => {
        const x = n.rx * w;
        const y = n.ry * h;
        const isFocus = idx === selectedIdx;
        const stepHit = Math.max(0, 1 - Math.abs(activeStep - idx) * 1.1);
        const haloR = n.r * (2.1 + stepHit * 1.4 + (isFocus ? 0.9 : 0));

        const halo = ctx.createRadialGradient(x, y, 2, x, y, haloR);
        halo.addColorStop(0, `${n.color}cc`);
        halo.addColorStop(0.5, `${n.color}33`);
        halo.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(x, y, haloR, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        ctx.arc(x, y, n.r * (isFocus ? 0.85 : 0.72), 0, Math.PI * 2);
        ctx.fill();

        if (isFocus) {
          ctx.strokeStyle = n.color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(x, y, n.r + 5, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.textAlign = 'center';
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.fillText(n.name, x, y + n.r + 18);
        ctx.fillStyle = n.color;
        ctx.font = '10px monospace';
        ctx.fillText(n.tag, x, y + n.r + 31);

        const nodeMs = getStarMs(n.id);
        if (nodeMs !== null && nodeMs !== undefined) {
          ctx.fillStyle = '#cbd5e1';
          ctx.font = 'bold 10px monospace';
          ctx.fillText(`${Number(nodeMs).toFixed(2)} ms`, x, y + n.r + 44);
        }
      });

      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resize);
    };
  }, [status, latestDecision, selectedIdx]);

  const selected = STAR_NODES[selectedIdx];
  const tf = status?.candle_interval || '1d';
  const lat = status?.stage_latencies_ms ?? latestDecision?.qwen_json?.stage_latencies_ms;
  const totalCycleMs = lat?.total_cycle_ms ?? latestDecision?.latency_ms;
  const fbTotalMs =
    lat?.feedback_laya1_ms !== undefined || lat?.feedback_laya2_ms !== undefined
      ? (lat?.feedback_laya1_ms ?? 0) + (lat?.feedback_laya2_ms ?? 0)
      : undefined;

  return (
    <section className="rounded-xl border border-indigo-500/30 bg-slate-950 p-4 shadow-xl space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-indigo-200">
            ✦ Mapa de Constelaciones • Paso en Vivo de la Información ({tf === '1h' ? 'Hora 1h' : 'Día 1d'})
          </h3>
          <p className="text-xs text-slate-400">
            Flujo de señales, retroalimentación de Qwen (Laya #1/#2) y tiempos en milisegundos.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded border border-indigo-500/30 bg-slate-900 px-2 py-1 font-mono text-[11px] text-indigo-200">
            Ciclo IA: {totalCycleMs !== undefined ? `${Number(totalCycleMs).toFixed(2)} ms` : '-- ms'}
          </span>
          <span className="rounded border border-amber-500/30 bg-slate-900 px-2 py-1 font-mono text-[11px] text-amber-300">
            Feedback Qwen: {fbTotalMs !== undefined ? `${fbTotalMs.toFixed(2)} ms` : '-- ms'}
          </span>
          <button
            type="button"
            onClick={() => {
              pulseStartRef.current = performance.now();
            }}
            className="rounded-lg border border-indigo-500/40 bg-indigo-500/20 px-3 py-1.5 text-xs font-semibold text-indigo-200 hover:bg-indigo-500/30"
          >
            ✨ Lanzar Pulso
          </button>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-indigo-950 bg-[#020410]" style={{ height: 420 }}>
        <canvas
          ref={canvasRef}
          onClick={(e) => {
            const canvas = canvasRef.current;
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            STAR_NODES.forEach((n, idx) => {
              if (Math.hypot(mx - n.rx * canvas.width, my - n.ry * canvas.height) <= 26) {
                setSelectedIdx(idx);
              }
            });
          }}
          className="block h-full w-full cursor-pointer"
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-indigo-500/20 bg-slate-900/70 px-3 py-2 text-xs">
        <span className="font-bold text-white">★ {selected.name} ({selected.tag})</span>
        <span className="font-mono text-[11px] text-slate-300">
          Ticker: {status?.active_ticker || 'BTC-USD'} | Última acción: {latestDecision?.final_action || 'HOLD'} | Estado: {status?.system_status || 'RUNNING'}
        </span>
      </div>
    </section>
  );
};
