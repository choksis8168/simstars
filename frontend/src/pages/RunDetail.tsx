import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type RunOut } from '../api'

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const [run, setRun] = useState<RunOut | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId) return
    api.getRun(runId).then(setRun).catch((err) => setError(err.message))
  }, [runId])

  if (error) return <p className="error">{error}</p>
  if (!run) return <p>Loading...</p>

  return (
    <div className="page">
      <Link to={`/sessions/${run.session_id}`}>&larr; Back to session</Link>
      <h1>Run</h1>
      <p className="card-meta">
        {run.end_reason ?? 'in progress'} &middot; {run.critic_attempts} critic attempt(s)
        {run.branch_rounds_used > 0 ? ` · ${run.branch_rounds_used} segment re-preview(s)` : ''}
      </p>
      {run.producer_note && <p className="forcing-mechanic">Producer's note: {run.producer_note}</p>}

      {run.audio_url && (
        <audio controls src={run.audio_url} className="audio-player">
          Your browser doesn't support inline audio playback -{' '}
          <a href={run.audio_url}>download the movie</a> instead.
        </audio>
      )}

      {run.scenes.length === 0 && <p>No screenplay yet.</p>}
      {run.scenes.map((scene, i) => (
        <div className="scene" key={i}>
          <h3>{scene.heading}</h3>
          {scene.sfx_cues.length > 0 && <p className="cue">SFX: {scene.sfx_cues.join(', ')}</p>}
          {scene.music_cue && <p className="cue">Music: {scene.music_cue}</p>}
          {scene.lines.map((line, j) => (
            <p className="line" key={j}>
              {line}
            </p>
          ))}
        </div>
      ))}

      {run.critic_reasoning && (
        <details className="critic-reasoning">
          <summary>Critic reasoning</summary>
          <p>{run.critic_reasoning}</p>
        </details>
      )}
    </div>
  )
}
