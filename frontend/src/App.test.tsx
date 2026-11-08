import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import { HomePage } from './pages/HomePage'
import { SearchPage } from './pages/SearchPage'

const api = vi.hoisted(() => ({
  getHealth: vi.fn(),
  getLanguages: vi.fn(),
  getTechnologies: vi.fn(),
  search: vi.fn(),
}))
vi.mock('./services/api', () => api)

function withRouter(node: React.ReactNode) {
  return <BrowserRouter>{node}</BrowserRouter>
}

describe('frontend API integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.pushState({}, '', '/')
    api.getHealth.mockResolvedValue({ status: 'ok', lexical_index: true, semantic_index: true, documents: 2725, version: '1.0' })
    api.getLanguages.mockResolvedValue([{ name: 'python', count: 10 }])
    api.getTechnologies.mockResolvedValue([{ name: 'django', count: 4 }])
    api.search.mockResolvedValue({ query: 'TypeError', mode: 'hybrid_rrf', total: 1, metadata: {}, results: [{ doc_id: 'so-1', title: 'TypeError fix', error_message: 'TypeError', description: 'description', solution: 'solution', code_example: null, tags: ['python'], language: 'python', technologies: ['django'], source_url: 'https://stackoverflow.com/questions/1', source_site: 'stackoverflow', score: 0.9, ranking_explanation: { reason: 'matched' }, snippet: 'TypeError', highlighted_snippet: '<mark>TypeError</mark>' }] })
  })

  it('submits the homepage query and navigates to search', () => {
    render(withRouter(<HomePage />))
    const input = screen.getByLabelText(/search programming errors/i)
    fireEvent.change(input, { target: { value: 'TypeError' } })
    fireEvent.submit(input.closest('form')!)
    expect(window.location.pathname).toBe('/search')
    expect(window.location.search).toContain('q=TypeError')
  })

  it('renders live results returned by the API', async () => {
    window.history.pushState({}, '', '/search?q=TypeError')
    render(withRouter(<SearchPage />))
    expect(await screen.findByText('TypeError fix')).toBeInTheDocument()
    expect(screen.getByText('Why this result?')).toBeInTheDocument()
    expect(api.search).toHaveBeenCalledWith(expect.objectContaining({ q: 'TypeError' }), expect.any(AbortSignal))
  })

  it('shows a backend failure state', async () => {
    api.search.mockRejectedValueOnce(new Error('Search indexes are unavailable'))
    window.history.pushState({}, '', '/search?q=TypeError')
    render(withRouter(<SearchPage />))
    expect(await screen.findByText('Search unavailable')).toBeInTheDocument()
    expect(screen.getByText('Search indexes are unavailable')).toBeInTheDocument()
  })
})
