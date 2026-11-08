import type {
  DocumentDetail,
  EvaluationResponse,
  Facet,
  HealthResponse,
  SearchParams,
  SearchResponse,
  SimilarResponse,
} from '../types/api'

const API_ROOT = import.meta.env.VITE_API_ROOT ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = (await response.json()) as { detail?: string }
      message = body.detail || message
    } catch {
      // Preserve the HTTP status message when the response is not JSON.
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export function search(params: SearchParams, signal?: AbortSignal): Promise<SearchResponse> {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== '') query.set(key, String(value))
  })
  return request<SearchResponse>(`/search?${query.toString()}`, { signal })
}

export function getDocument(id: string, signal?: AbortSignal): Promise<DocumentDetail> {
  return request<DocumentDetail>(`/documents/${encodeURIComponent(id)}`, { signal })
}

export function getSimilar(id: string, signal?: AbortSignal): Promise<SimilarResponse> {
  return request<SimilarResponse>(`/similar/${encodeURIComponent(id)}`, { signal })
}

export function getLanguages(signal?: AbortSignal): Promise<Facet[]> {
  return request<Facet[]>('/languages', { signal })
}

export function getTechnologies(signal?: AbortSignal): Promise<Facet[]> {
  return request<Facet[]>('/technologies', { signal })
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

export function getEvaluation(signal?: AbortSignal): Promise<EvaluationResponse> {
  return request<EvaluationResponse>('/evaluation', { signal })
}
