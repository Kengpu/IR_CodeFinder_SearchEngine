import { useEffect, useState } from 'react'
import { ArrowLeft, Check, Copy, ExternalLink, Sparkles } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { getDocument } from '../services/api'
import type { DocumentDetail } from '../types/api'

export function DocumentPage() {
  const { id = '' } = useParams()
  const [document, setDocument] = useState<DocumentDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getDocument(id, controller.signal).then(setDocument).catch((reason: unknown) => { if ((reason as Error).name !== 'AbortError') setError((reason as Error).message) }).finally(() => setLoading(false))
    return () => controller.abort()
  }, [id])
  async function copyCode() { if (!document?.code_example) return; await navigator.clipboard.writeText(document.code_example); setCopied(true); window.setTimeout(() => setCopied(false), 1800) }
  if (loading) return <main className="detail-page"><div className="state-card"><div className="spinner" /> Loading document…</div></main>
  if (error || !document) return <main className="detail-page"><div className="state-card error-state"><strong>Document unavailable</strong><p>{error || 'This document could not be found.'}</p><Link to="/">Return home</Link></div></main>
  return <main className="detail-page">
    <Link className="back-link" to="/search"><ArrowLeft size={16} /> Back to results</Link>
    <div className="detail-grid"><article className="detail-main"><div className="detail-meta"><span className="result-language">{document.language || 'general'}</span>{document.technologies.map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div><h1>{document.title}</h1>{document.error_message && <div className="detail-error"><span>ERROR SIGNATURE</span><code>{document.error_message}</code></div>}<section><span className="section-kicker">THE PROBLEM</span><p className="detail-copy">{document.description}</p></section><section><span className="section-kicker">SOLUTION</span><p className="detail-copy">{document.solution}</p></section>{document.code_example && <section className="code-section"><div className="code-heading"><span className="section-kicker">CODE EXAMPLE</span><button onClick={copyCode}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy code'}</button></div><pre><code>{document.code_example}</code></pre></section>}<a className="source-button" href={document.source_url} target="_blank" rel="noreferrer">View original on {document.source_site} <ExternalLink size={15} /></a><div className="attribution-note"><span>Question license · {document.license}</span>{document.answer_license && <span>Answer license · {document.answer_license}</span>}<div>{document.source_author && <span>Question by {document.source_author_url ? <a href={document.source_author_url} target="_blank" rel="noreferrer">{document.source_author}</a> : document.source_author}</span>}{document.answer_author && <span>Solution by {document.answer_author_url ? <a href={document.answer_author_url} target="_blank" rel="noreferrer">{document.answer_author}</a> : document.answer_author}</span>}</div><small>Content is reproduced under the source license. Please follow the original attribution and share-alike terms.</small></div></article><aside className="detail-aside"><div className="explanation-card"><div className="aside-title"><Sparkles size={15} /> Why this result?</div><p>{document.ranking_explanation.reason || 'This document was retrieved from the indexed corpus.'}</p><div className="detail-score"><span>Final ranking score</span><strong>{document.score.toFixed(4)}</strong></div>{Object.entries(document.ranking_explanation).filter(([key, value]) => key.endsWith('_score') && typeof value === 'number').map(([key, value]) => <div className="metric-row" key={key}><span>{key.replaceAll('_', ' ')}</span><span>{Number(value).toFixed(4)}</span></div>)}</div>{document.related_errors.length > 0 && <div className="related-card"><div className="aside-title">Related errors</div>{document.related_errors.slice(0, 4).map((related) => <Link key={related.doc_id} to={`/documents/${related.doc_id}`}><span>{related.title}</span><small>{related.score.toFixed(3)}</small></Link>)}</div>}</aside></div>
  </main>
}
