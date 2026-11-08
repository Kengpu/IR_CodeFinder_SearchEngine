import { useEffect, useState } from 'react'
import { getHealth } from '../services/api'
import type { HealthResponse } from '../types/api'

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal).then(setHealth).catch(() => setHealth(null))
    return () => controller.abort()
  }, [])
  return health
}
