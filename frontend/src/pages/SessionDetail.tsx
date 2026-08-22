import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, type JobKind, type SessionDetailOut } from '../api'
import { usePollJob } from '../usePollJob'

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const [session, setSession] = useState<SessionDetailOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [activeJobKind, setActiveJobKind] = useState<JobKind | null>(null)

  const load = useCallback(() => {
    if (!sessionId) return
    api.getSession(sessionId).then(setSession).catch((err) => setError(err.message))
  }, [sessionId])

  useEffect(load, [load])

  const job = usePollJob(activeJobId)

  // Once a job finishes: refresh the session (new run appears in history)
  // and jump straight to the run's page.
  useEffect(() => {
    if (!job) return
    if (job.status === 'complete' && job.result_run_id) {
      setActiveJobId(null)
      navigate(`/runs/${job.result_run_id}`)
    } else if (job.status === 'failed') {
      setActiveJobId(null)
      load()
    }
  }, [job, navigate, load])

  async function trigger(kind: JobKind) {
    if (!sessionId) return
    setError(null)
    try {
      const started = kind === 'generate' ? await api.startGenerate(sessionId, note) : await api.startPlay(sessionId, note)
      setActiveJobKind(kind)
      setActiveJobId(started.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  if (error && !session) return <p className="error">{error}</p>
  if (!session) return <p>Loading...</p>

  const busy = activeJobId !== null && job?.status !== 'complete' && job?.status !== 'failed'

  return (
    <div className="page">
      <Link to="/">&larr; All sessions</Link>
      <h1>{session.world_description}</h1>
      {session.forcing_mechanic && <p className="forcing-mechanic">{session.forcing_mechanic}</p>}
      <p className="card-meta">Locations: {session.locations.join(', ')}</p>

      <h2>Cast</h2>
      <ul className="card-list">
        {session.characters.map((c) => (
          <li key={c.id} className="card static">
            <div className="card-title">
              {c.name} &middot; {c.role}
            </div>
            <div className="card-meta">
              {c.traits} - starts at {c.starting_location}
            </div>
          </li>
        ))}
      </ul>

      <h2>Make a movie</h2>
      <label className="field">
        <span>Producer's note (optional) - nudges the next run without dictating it</span>
        <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. Ana should crack first" />
      </label>
      <div className="row">
        <button className="button" onClick={() => trigger('generate')} disabled={busy}>
          Generate Script (free)
        </button>
        <button className="button" onClick={() => trigger('play')} disabled={busy}>
          Play Movie
        </button>
      </div>

      {busy && (
        <p className="status">
          {activeJobKind === 'generate' ? 'Simulating' : 'Simulating and producing'}... ({job?.status ?? 'starting'})
        </p>
      )}
      {job?.status === 'failed' && <p className="error">{job.error_message}</p>}
      {error && <p className="error">{error}</p>}

      <h2>Past runs</h2>
      {session.runs.length === 0 && <p>No runs yet.</p>}
      <ul className="card-list">
        {session.runs.map((r) => (
          <li key={r.id} className="card">
            <Link to={`/runs/${r.id}`}>
              <div className="card-title">{new Date(r.created_at).toLocaleString()}</div>
              <div className="card-meta">
                {r.end_reason ?? 'in progress'} &middot; {r.critic_attempts} attempt(s)
                {r.audio_url ? ' · has audio' : ''}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
