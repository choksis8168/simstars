import { useEffect, useState } from 'react'
import { api, type JobOut } from './api'

const POLL_INTERVAL_MS = 2000

/** Polls /api/jobs/{id} every couple seconds until the job reaches a
 * terminal status (complete/failed) - the polling approach chosen for
 * generate/play progress instead of websockets, see docs/design.md web-app
 * plan. Returns null until the first poll resolves or if jobId is null. */
export function usePollJob(jobId: string | null): JobOut | null {
  const [job, setJob] = useState<JobOut | null>(null)

  useEffect(() => {
    setJob(null)
    if (!jobId) return

    let cancelled = false
    let timer: number | undefined

    async function poll() {
      try {
        const result = await api.getJob(jobId!)
        if (cancelled) return
        setJob(result)
        if (result.status === 'pending' || result.status === 'running') {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS)
        }
      } catch (err) {
        if (!cancelled) console.error('job poll failed', err)
      }
    }
    poll()

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [jobId])

  return job
}
