import { Code2 } from 'lucide-react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { useHealth } from '../hooks/useHealth'

export function Layout() {
  const health = useHealth()
  return <div className="app-shell"><header className="topbar"><Link to="/" className="brand"><span className="brand-mark"><Code2 size={19} /></span><span>Code<span>Finder</span></span></Link><nav><NavLink to="/" end>Search</NavLink><NavLink to="/evaluation">Evaluation</NavLink><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API docs</a></nav><div className="topbar-status"><span className={`status-indicator ${health?.lexical_index ? 'online' : ''}`} />{health?.lexical_index ? 'Index online' : 'Connecting'}</div></header><Outlet /><footer className="footer"><span>CodeFinder <span className="footer-dot">·</span> Developer Error Search</span><span>Real Q&amp;A · Traceable sources · Open search</span></footer></div>
}
