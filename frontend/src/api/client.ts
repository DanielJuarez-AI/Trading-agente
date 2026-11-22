/**
 * Cliente HTTP del Frontend React.
 * SEGURIDAD: NUNCA almacena GATEWAY_API_KEY ni secretos del servidor en variables VITE_*.
 * Las acciones protegidas (/api/control/*) se autentican mediante sesión HttpOnly
 * emitida por el backend al ingresar la credencial del operador en tiempo de ejecución.
 */

export const OFFICIAL_PRIMARY_TICKERS: string[] = [
  'AAPL', 'MSFT', 'NVDA', 'TSLA', 'HWM', 'JPM', 'GOOGL', 'AMZN', 'META', 'LMT',
  'WM', 'LULU', 'DIS', 'V', 'MA', 'BTC-USD', 'SPY', 'QQQ', 'NFLX', 'AMD',
  'CRM', 'ADBE', 'PYPL', 'COST', 'HD', 'UNH', 'JNJ', 'PG', 'KO', 'WMT',
];

export interface AgentNodeMetadata {
  execution_mode: 'EXTERNAL_API' | 'LOCAL_DEVICE';
  is_local_device: boolean;
  location_label: string;
  endpoint: string;
  model: string;
}

export interface SystemStatusResponse {
  system_status: 'RUNNING' | 'PAUSED' | 'STOPPED';
  active_ticker: string;
  source_mode: 'live' | 'replay';
  simulation_profile: string;
  profile_config: {
    name: string;
    initial_capital: number;
    fee_pct: number;
  };
  enable_real_trading: boolean;
  primary_tickers: string[];
  allowed_symbols: string[];
  hybrid_topology: {
    qwen_orchestrator: AgentNodeMetadata;
    laya_alpha: AgentNodeMetadata;
    laya_guard: AgentNodeMetadata;
  };
  portfolio: {
    cash: number;
    position_qty: number;
    avg_entry_price: number;
    last_price: number;
    total_equity: number;
    peak_equity: number;
    drawdown_pct: number;
    daily_pnl_pct: number;
  };
  metrics: {
    initial_capital: number;
    current_equity: number;
    net_return_pct: number;
    max_drawdown_pct: number;
    win_rate: number | string;
    closed_trades: number;
    total_orders: number;
  };
  processed_candles_count: number;
}

export interface DecisionRecord {
  id: number;
  symbol: string;
  interval: string;
  candle_close_utc: string;
  model_pred: number;
  laya1_json: {
    regime?: string;
    signal_action?: string;
    conviction_level?: string;
    conviction_score?: number;
    execution_mode?: string;
    is_local_device?: boolean;
    transport?: string;
    latency_ms?: number;
  };
  qwen_json: {
    action?: string;
    symbol?: string;
    candle_date?: string;
    justification?: string;
    execution_mode?: string;
    is_local_device?: boolean;
    api_endpoint?: string;
    latency_ms?: number;
  };
  laya2_json: {
    approved?: boolean;
    security_status?: string;
    coherence_status?: string;
    reason?: string;
    execution_mode?: string;
    is_local_device?: boolean;
    latency_ms?: number;
  };
  risk_status: 'APPROVED' | 'REJECTED';
  risk_reason: string;
  final_action: 'BUY' | 'HOLD' | 'SELL';
  latency_ms: number;
  qwen_execution_mode: string;
  laya_execution_mode: string;
  source_mode: 'live' | 'replay';
  created_at: string;
}

export interface ErrorLogRecord {
  id: number;
  error_code: string;
  component: string;
  message: string;
  payload_excerpt: string;
  created_at: string;
}

export interface BenchmarkLane {
  strategy: string;
  initial_capital: number;
  final_equity: number;
  net_return_pct: number;
  max_drawdown_pct: number;
  win_rate: number | string;
  closed_trades: number;
  total_orders: number;
  avg_latency_ms: number;
  rejected_actions: number;
}

export interface BenchmarkResponse {
  symbol: string;
  cycles_evaluated: number;
  profile: string;
  lanes: BenchmarkLane[];
  interpretation: string;
}

export interface CandlePoint {
  time: string;
  open_time: string;
  close_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ema10: number;
  ema20: number;
}

export interface CandleMarker {
  time: string;
  action: 'BUY' | 'SELL';
  risk_status: string;
  model_pred: number;
}

export interface CandlesResponse {
  symbol: string;
  data_source: string;
  last_close: number;
  prev_close: number;
  change_pct_1d: number;
  high_period: number;
  low_period: number;
  candles: CandlePoint[];
  markers: CandleMarker[];
}

const API_BASE = '';

async function requestJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = await res.json();
      detail = errBody.detail || detail;
    } catch {
      // ignore parse failure
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const apiClient = {
  checkSession: () => requestJson<{ authenticated: boolean }>('/api/auth/session'),
  login: (credential: string) =>
    requestJson<{ authenticated: boolean; message: string }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ credential }),
    }),
  logout: () =>
    requestJson<{ authenticated: boolean }>('/api/auth/logout', { method: 'POST' }),
  getStatus: () => requestJson<SystemStatusResponse>('/api/status'),
  getDecisions: (limit = 50) =>
    requestJson<{ items: DecisionRecord[] }>(`/api/decisions?limit=${limit}`),
  getErrors: (limit = 50) =>
    requestJson<{ items: ErrorLogRecord[] }>(`/api/errors?limit=${limit}`),
  getCandles: (symbol?: string, limit = 140) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (symbol) params.set('symbol', symbol);
    return requestJson<CandlesResponse>(`/api/candles?${params.toString()}`);
  },
  getBenchmark: () => requestJson<BenchmarkResponse>('/api/benchmark'),
  control: (
    action: 'start' | 'pause' | 'stop' | 'step' | 'replay' | 'switch_ticker' | 'switch_mode' | 'benchmark',
    payload?: { symbol?: string; mode?: 'live' | 'replay'; n_cycles?: number }
  ) =>
    requestJson<Record<string, unknown>>(`/api/control/${action}`, {
      method: 'POST',
      body: JSON.stringify(payload || {}),
    }),
};
