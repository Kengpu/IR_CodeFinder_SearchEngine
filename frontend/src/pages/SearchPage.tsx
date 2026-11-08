import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, ChevronDown, Clock3, ExternalLink, Filter, Search, SlidersHorizontal, Sparkles, X } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { getLanguages, getTechnologies, search } from '../services/api'
import type { Facet, SearchDocument, SearchMode, SearchResponse } from '../types/api'

const modes: Array<{ value: SearchMode; label: string; description: string }> = [
  { value: 'hybrid_rrf', label: 'Hybrid RRF', description: 'Lexical + semantic, fused by rank' },
  { value: 'hybrid_weighted', label: 'Hybrid weighted', description: 'Weighted score fusion' },
  { value: 'hybrid_prf', label: 'Hybrid + PRF', description: 'Pseudo-relevance feedback' },
  { value: 'bm25', label: 'BM25', description: 'Probabilistic lexical ranking' },
  { value: 'tfidf', label: 'TF-IDF', description: 'Term-frequency ranking' },
  { value: 'semantic', label: 'Semantic', description: 'Meaning-based vector search' },
  { value: 'boolean', label: 'Boolean', description: 'AND / OR / NOT matching' },
]

function ResultCard({ document, position }: { document: SearchDocument; position: number }) {
  const [expanded, setExpanded] = useState(false)
  return <article className="result-card">
    <div className="result-rank">{String(position).padStart(2, '0')}</div>
    <div className="result-body">
      <div className="result-topline"><span className="result-language">{document.language || 'general'}</span><span className="result-score">score {document.score.toFixed(3)}</span></div>
      <Link className="result-title" to={`/documents/${encodeURIComponent(document.doc_id)}`}>{document.title}</Link>
      {document.error_message && <code className="error-line">{document.error_message}</code>}
      <p className="result-snippet" dangerouslySetInnerHTML={{ __html: document.highlighted_snippet || document.snippet }} />
      <div className="result-footer">
        <div className="badge-list">{document.technologies.slice(0, 4).map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div>
        <div className="result-actions"><button onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide why' : 'Why this result?'} <ChevronDown size={14} className={expanded ? 'rotate' : ''} /></button><a href={document.source_url} target="_blank" rel="noreferrer">Source <ExternalLink size={13} /></a></div>
      </div>
      {expanded && <div className="ranking-panel"><div className="ranking-panel-title"><Sparkles size={14} /> Ranking explanation</div><p>{document.ranking_explanation.reason || 'Ranked by the selected retrieval method.'}</p><div className="score-grid">{Object.entries(document.ranking_explanation).filter(([key, value]) => key.endsWith('_score') && typeof value === 'number').slice(0, 4).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{Number(value).toFixed(4)}</strong></div>)}</div></div>}
    </div>
  </article>
}

export function SearchPage() {
  const [params, setParams] = useSearchParams()
  const query = params.get('q') || ''
  const [input, setInput] = useState(query)
  const [mode, setMode] = useState<SearchMode>((params.get('mode') as SearchMode) || 'hybrid_rrf')
  const [language, setLanguage] = useState(params.get('language') || '')
  const [technology, setTechnology] = useState(params.get('technology') || '')
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [languages, setLanguages] = useState<Facet[]>([])
  const [technologies, setTechnologies] = useState<Facet[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => { getLanguages().then(setLanguages).catch(() => undefined); getTechnologies().then(setTechnologies).catch(() => undefined) }, [])
  useEffect(() => {
    if (!query) return
    const controller = new AbortController()
    setLoading(true); setError('')
    search({ q: query, mode, language: language || undefined, technology: technology || undefined, top_k: 20 }, controller.signal).then(setResponse).catch((reason: unknown) => { if ((reason as Error).name !== 'AbortError') setError((reason as Error).message) }).finally(() => setLoading(false))
    return () => controller.abort()
  }, [query, mode, language, technology])

  function update(next: Partial<{ q: string; mode: SearchMode; language: string; technology: string }>) {
    const nextParams = new URLSearchParams(params)
    Object.entries(next).forEach(([key, value]) => value ? nextParams.set(key, value) : nextParams.delete(key))
    setParams(nextParams)
  }
  const activeFilters = useMemo(() => [language && `Language: ${language}`, technology && `Technology: ${technology}`].filter(Boolean) as string[], [language, technology])

  return <main className="search-page">
    <div className="search-toolbar"><Link className="back-link" to="/"><ArrowLeft size={16} /> Home</Link><form className="compact-search" onSubmit={(event) => { event.preventDefault(); update({ q: input }) }}><Search size={17} /><input aria-label="Search" value={input} onChange={(event) => setInput(event.target.value)} /><button type="submit">Search</button></form></div>
    <div className="search-layout">
      <aside className={`filters ${showFilters ? 'open' : ''}`}><div className="filter-heading"><span><Filter size={16} /> Refine results</span><button onClick={() => setShowFilters(false)} aria-label="Close filters"><X size={17} /></button></div><label>Search mode<select value={mode} onChange={(event) => { const value = event.target.value as SearchMode; setMode(value); update({ mode: value }) }}>{modes.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select><small>{modes.find((item) => item.value === mode)?.description}</small></label><label>Language<select value={language} onChange={(event) => { setLanguage(event.target.value); update({ language: event.target.value }) }}><option value="">All languages</option>{languages.map((facet) => <option value={facet.name} key={facet.name}>{facet.name} ({facet.count})</option>)}</select></label><label>Technology<select value={technology} onChange={(event) => { setTechnology(event.target.value); update({ technology: event.target.value }) }}><option value="">All technologies</option>{technologies.map((facet) => <option value={facet.name} key={facet.name}>{facet.name} ({facet.count})</option>)}</select></label>{activeFilters.length > 0 && <button className="clear-filters" onClick={() => { setLanguage(''); setTechnology(''); update({ language: '', technology: '' }) }}>Clear filters</button>}</aside>
      <section className="results-column"><div className="results-heading"><div><span className="section-kicker">SEARCH RESULTS</span><h1>{query || 'Search the corpus'}</h1>{response && <p><strong>{response.total}</strong> results · {modes.find((item) => item.value === response.mode)?.label}</p>}</div><button className="mobile-filter-button" onClick={() => setShowFilters(true)}><SlidersHorizontal size={16} /> Filters</button></div>{activeFilters.length > 0 && <div className="active-filters">{activeFilters.map((filter) => <span key={filter}>{filter}</span>)}<Clock3 size={14} /> live index</div>}{loading && <div className="state-card"><div className="spinner" /> Searching the indexed corpus…</div>}{error && <div className="state-card error-state"><X size={22} /><strong>Search unavailable</strong><p>{error}</p></div>}{!loading && !error && response && response.results.length === 0 && <div className="state-card"><Search size={25} /><strong>No matching documents</strong><p>Try an exact error name, a broader question, or remove a filter.</p></div>}{!loading && !error && response?.results.map((document, index) => <ResultCard key={document.doc_id} document={document} position={index + 1} />)}</section>
    </div>
  </main>
}
