import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type SessionOut } from '../api'

export function SessionList() {
  const [sessions, setSessions] = useState<SessionOut[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.listSessions().then(setSessions).catch((err) => setError(err.message))
  }, [])

  return (
    <div className="page">
      <div className="page-header">
        <h1>SimStars</h1>
        <Link className="button" to="/new">
          + New Session
        </Link>
      </div>
      <p className="tagline">Create the characters. Set the world. Press play. Watch a movie emerge.</p>

      {error && <p className="error">{error}</p>}
      {sessions === null && !error && <p>Loading...</p>}
      {sessions?.length === 0 && <p>No sessions yet - create one to get started.</p>}

      <ul className="card-list">
        {sessions?.map((s) => (
          <li key={s.id} className="card">
            <Link to={`/sessions/${s.id}`}>
              <div className="card-title">{s.world_description}</div>
              <div className="card-meta">
                {s.characters.map((c) => c.name).join(', ')} · {s.locations.join(', ')}
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}
