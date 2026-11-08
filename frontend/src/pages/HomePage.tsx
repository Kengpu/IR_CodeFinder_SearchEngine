import { useState } from 'react'
import { ArrowRight, Code2, Search, Sparkles, Terminal } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useHealth } from '../hooks/useHealth'

const examples = [
  'TypeError: cannot read properties of undefined',
  'Python ModuleNotFoundError: No module named pandas',
  'npm install ERESOLVE unable to resolve dependency tree',
]
const popular = ['python', 'javascript', 'reactjs', 'django', 'pandas', 'mysql', 'docker']

export function HomePage() {
  const [query, setQuery] = useState('')
  const health = useHealth()
  const navigate = useNavigate()

  function submit(value = query) {
    const trimmed = value.trim()
    if (trimmed) navigate(`/search?q=${encodeURIComponent(trimmed)}`)
  }

  return (
    <main className="home-page">
      <section className="hero">
        <div className="eyebrow"><span className="pulse-dot" /> Developer knowledge, indexed</div>
        <h1>Find the fix.<br /><span>Understand the error.</span></h1>
        <p className="hero-copy">Search real developer questions and solutions across languages, frameworks, and tools.</p>
        <form className="hero-search" onSubmit={(event) => { event.preventDefault(); submit() }}>
          <Search size={21} aria-hidden="true" />
          <input
            aria-label="Search programming errors and questions"
            placeholder="Paste an error or describe your problem…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="submit">Search <ArrowRight size={16} /></button>
        </form>
        <div className="search-hint"><kbd>↵</kbd> to search <span>•</span> Exact errors and natural-language questions</div>
        <div className="example-row">
          <span>Try:</span>
          {examples.map((example) => <button key={example} onClick={() => { setQuery(example); submit(example) }}>{example}</button>)}
        </div>
      </section>

      <section className="popular-section" aria-labelledby="popular-title">
        <div className="section-heading">
          <div><span className="section-kicker">START EXPLORING</span><h2 id="popular-title">Popular technologies</h2></div>
          <span className="section-note">Filters are drawn from the live corpus</span>
        </div>
        <div className="technology-grid">
          {popular.map((technology, index) => (
            <button className="technology-card" key={technology} onClick={() => submit(`error ${technology}`)}>
              <span className={`tech-icon tech-${index}`}><Code2 size={19} /></span>
              <span className="tech-name">{technology}</span>
              <ArrowRight size={15} className="tech-arrow" />
            </button>
          ))}
        </div>
      </section>

      <section className="feature-strip">
        <div className="feature-item"><span className="feature-icon"><Terminal size={18} /></span><div><strong>Built for real errors</strong><span>Source-linked answers from developer Q&A</span></div></div>
        <div className="feature-item"><span className="feature-icon"><Sparkles size={18} /></span><div><strong>Search by meaning</strong><span>Lexical, semantic, or hybrid retrieval</span></div></div>
        <div className="feature-status"><span className={`status-indicator ${health?.lexical_index ? 'online' : ''}`} />{health?.lexical_index ? `${health.documents.toLocaleString()} indexed documents` : 'Connecting to search index'}</div>
      </section>
    </main>
  )
}
