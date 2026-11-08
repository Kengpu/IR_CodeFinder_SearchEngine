import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { DocumentPage } from './pages/DocumentPage'
import { EvaluationPage } from './pages/EvaluationPage'
import { HomePage } from './pages/HomePage'
import { SearchPage } from './pages/SearchPage'

export function App() {
  return <BrowserRouter><Routes><Route element={<Layout />}><Route path="/" element={<HomePage />} /><Route path="/search" element={<SearchPage />} /><Route path="/documents/:id" element={<DocumentPage />} /><Route path="/evaluation" element={<EvaluationPage />} /></Route></Routes></BrowserRouter>
}
