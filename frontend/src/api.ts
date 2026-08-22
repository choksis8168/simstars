// Thin fetch() wrapper matching the FastAPI backend's response shapes
// (see src/simstars/api.py) exactly - one file, kept in sync by hand since
// this is a small local app, not generated from an OpenAPI spec.

export interface CharacterOut {
  id: string
  name: string
  role: string
  traits: string
  starting_location: string
  voice_id: string | null
}

export interface SessionOut {
  id: string
  created_at: string
  world_description: string
  forcing_mechanic: string | null
  locations: string[]
  characters: CharacterOut[]
}

export interface SceneOut {
  location: string
  heading: string
  lines: string[]
  sfx_cues: string[]
  music_cue: string | null
}

export interface RunOut {
  id: string
  session_id: string
  created_at: string
  producer_note: string | null
  end_reason: string | null
  critic_attempts: number
  critic_reasoning: string | null
  branch_rounds_used: number
  scenes: SceneOut[]
  audio_url: string | null
}

export interface SessionDetailOut extends SessionOut {
  runs: RunOut[]
}

export type JobKind = 'generate' | 'play'
export type JobStatus = 'pending' | 'running' | 'complete' | 'failed'

export interface JobOut {
  id: string
  session_id: string
  kind: JobKind
  status: JobStatus
  error_message: string | null
  result_run_id: string | null
}

export interface CharacterIn {
  name: string
  role: string
  traits: string
  starting_location: string
}

export interface SessionCreateIn {
  world_description: string
  locations: string[]
  characters: CharacterIn[]
}

class ApiError extends Error {}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(body.detail || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  listSessions: () => request<SessionOut[]>('/sessions'),
  getSession: (id: string) => request<SessionDetailOut>(`/sessions/${id}`),
  createSession: (body: SessionCreateIn) =>
    request<SessionOut>('/sessions', { method: 'POST', body: JSON.stringify(body) }),
  startGenerate: (sessionId: string, note?: string) =>
    request<JobOut>(`/sessions/${sessionId}/generate`, { method: 'POST', body: JSON.stringify({ note }) }),
  startPlay: (sessionId: string, note?: string) =>
    request<JobOut>(`/sessions/${sessionId}/play`, { method: 'POST', body: JSON.stringify({ note }) }),
  getJob: (id: string) => request<JobOut>(`/jobs/${id}`),
  getRun: (id: string) => request<RunOut>(`/runs/${id}`),
}

export { ApiError }
