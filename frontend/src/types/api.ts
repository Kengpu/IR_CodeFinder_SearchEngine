export type SearchMode =
  | 'tfidf'
  | 'bm25'
  | 'boolean'
  | 'semantic'
  | 'hybrid_rrf'
  | 'hybrid_weighted'
  | 'hybrid_prf'

export interface RankingExplanation {
  method?: string
  reason?: string
  final_score?: number
  lexical_score?: number
  semantic_score?: number
  rrf_score?: number
  normalized_lexical_score?: number
  normalized_semantic_score?: number
  lexical_weight?: number
  semantic_weight?: number
  [key: string]: unknown
}

export interface SearchDocument {
  doc_id: string
  title: string
  error_message: string | null
  description: string
  solution: string
  code_example: string | null
  tags: string[]
  language: string | null
  technologies: string[]
  source_url: string
  source_site: string
  source_author: string | null
  source_author_url: string | null
  answer_author: string | null
  answer_author_url: string | null
  license: string
  answer_license: string | null
  score: number
  ranking_explanation: RankingExplanation
  snippet: string
  highlighted_snippet: string
}

export interface SearchResponse {
  query: string
  mode: string
  results: SearchDocument[]
  total: number
  metadata: Record<string, unknown>
}

export interface SearchParams {
  q: string
  mode: SearchMode
  top_k?: number
  language?: string
  technology?: string
  prf?: boolean
  rerank?: boolean
  lexical_weight?: number
  semantic_weight?: number
}

export interface SimilarResponse {
  document_id: string
  results: SearchDocument[]
  total: number
}

export interface DocumentDetail extends SearchDocument {
  related_errors: SearchDocument[]
}

export interface Facet {
  name: string
  count: number
}

export interface HealthResponse {
  status: string
  lexical_index: boolean
  semantic_index: boolean
  documents: number
  version: string
}

export interface EvaluationResponse {
  available: boolean
  metrics: Record<string, unknown>
  message: string | null
}
