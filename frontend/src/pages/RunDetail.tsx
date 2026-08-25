import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type RunOut } from '../api'
import { usePollJob } from '../usePollJob'

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const [run, setRun] = useState<RunOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [produceJobId, setProduceJobId] = useState<string | null>(null)

  const load = useCallback(() => {
    if (!runId) return
    api.getRun(runId).then(setRun).catch((err) => setError(err.message))
  }, [runId])

  useEffect(load, [load])

  const produceJob = usePollJob(produceJobId)

  // Once producing finishes: refresh so audio_url (or the failure) shows up.
  useEffect(() => {
    if (!produceJob) return
    if (produceJob.status === 'complete' || produceJob.status === 'failed') {
      setProduceJobId(null)
      load()
    }
  }, [produceJob, load])

  async function produceAudio() {
    if (!runId) return
    setError(null)
    try {
      const job = await api.startProduce(runId)
      setProduceJobId(job.id)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  if (error && !run) return <p className="error">{error}</p>
  if (!run) return <p>Loading...</p>

  const producing = produceJobId !== null && produceJob?.status !== 'complete' && produceJob?.status !== 'failed'

  return (
    <div className="page">
      <Link to={`/sessions/${run.session_id}`}>&larr; Back to session</Link>
      <h1>Run</h1>
      <p className="card-meta">
        {run.end_reason ?? 'in progress'} &middot; {run.critic_attempts} critic attempt(s)
        {run.branch_rounds_used > 0 ? ` · ${run.branch_rounds_used} segment re-preview(s)` : ''}
        {run.llm_calls > 0 ? ` · ${run.llm_calls} API calls (~$${run.estimated_cost_usd.toFixed(2)} est.)` : ''}
      </p>
      {run.producer_note && <p className="forcing-mechanic">Producer's note: {run.producer_note}</p>}

      {run.audio_url ? (
        <audio controls src={run.audio_url} className="audio-player">
          Your browser doesn't support inline audio playback -{' '}
          <a href={run.audio_url}>download the movie</a> instead.
        </audio>
      ) : (
        <div className="row">
          <button className="button" onClick={produceAudio} disabled={producing}>
            {producing ? `Producing audio... (${produceJob?.status ?? 'starting'})` : 'Produce Audio for This Run'}
          </button>
        </div>
      )}
      {produceJob?.status === 'failed' && <p className="error">{produceJob.error_message}</p>}
      {error && <p className="error">{error}</p>}

      {run.scenes.length === 0 && <p>No screenplay yet.</p>}
      {run.scenes.map((scene, i) => (
        <div className="scene" key={i}>
          <h3>{scene.heading}</h3>
          {scene.narration && <p className="narration">{scene.narration}</p>}
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
