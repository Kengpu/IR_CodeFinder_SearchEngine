import { useEffect, useState } from 'react'
import { Activity, ArrowLeft, BarChart3, Database, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { getEvaluation } from '../services/api'
import type { EvaluationResponse } from '../types/api'

interface MetricRow { name: string; ndcg: number; map: number; recall: number; latency: number }

export function EvaluationPage() {
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  function load(signal?: AbortSignal) {
    setLoading(true); setError('')
    getEvaluation(signal).then(setEvaluation).catch((reason: unknown) => { if ((reason as Error).name !== 'AbortError') setError((reason as Error).message) }).finally(() => setLoading(false))
  }
  useEffect(() => { const controller = new AbortController(); load(controller.signal); return () => controller.abort() }, [])
  const metrics = evaluation?.metrics as { metrics?: Record<string, Record<string, number>>; dataset_documents?: number; evaluation_query_count?: number; judgments?: string } | undefined
  const rows: MetricRow[] = Object.entries(metrics?.metrics || {}).map(([name, values]) => ({ name, ndcg: values['ndcg@10'] ?? 0, map: values.map ?? 0, recall: values['recall@10'] ?? 0, latency: values.latency_p95_ms ?? 0 }))
  const best = rows.reduce<MetricRow | null>((winner, row) => !winner || row.ndcg > winner.ndcg ? row : winner, null)
  return <main className="evaluation-page"><Link className="back-link" to="/"><ArrowLeft size={16} /> Home</Link><div className="evaluation-heading"><div><span className="section-kicker">MEASURED EXPERIMENTS</span><h1>Search evaluation</h1><p>Metrics are loaded from the backend's saved evaluation run.</p></div><button className="refresh-button" onClick={() => load()} disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} /> Refresh</button></div>
    {loading && <div className="state-card"><div className="spinner" /> Loading measured metrics…</div>}{error && <div className="state-card error-state"><strong>Evaluation unavailable</strong><p>{error}</p></div>}{!loading && !error && evaluation && !evaluation.available && <div className="state-card"><BarChart3 size={25} /><strong>No evaluation run found</strong><p>{evaluation.message || 'Run the evaluation script after building indexes and qrels.'}</p></div>}
    {!loading && evaluation?.available && <><div className="eval-kpis"><div><Database size={16} /><strong>{metrics?.dataset_documents?.toLocaleString() ?? '—'}</strong><span>Corpus documents</span></div><div><Activity size={16} /><strong>{metrics?.evaluation_query_count ?? '—'}</strong><span>Evaluation queries</span></div><div><BarChart3 size={16} /><strong>{best?.name ?? '—'}</strong><span>Best NDCG@10</span></div></div><div className="evaluation-card"><div className="evaluation-card-heading"><div><h2>Retrieval modes</h2><p>Actual measurements from the saved experiment</p></div></div><div className="evaluation-table-wrap"><table><thead><tr><th>Mode</th><th>NDCG@10</th><th>MAP</th><th>Recall@10</th><th>p95 (ms)</th></tr></thead><tbody>{rows.map((row) => <tr key={row.name}><td><span className="mode-name">{row.name.replaceAll('_', ' ')}</span>{row === best && <span className="best-tag">top NDCG</span>}</td><td>{row.ndcg.toFixed(4)}</td><td>{row.map.toFixed(4)}</td><td>{row.recall.toFixed(4)}</td><td>{row.latency.toFixed(2)}</td></tr>)}</tbody></table></div>{metrics?.judgments && <p className="eval-disclaimer">Judgment set: {metrics.judgments}. See the evaluation report for methodology and limitations.</p>}</div></>}
  </main>
}
