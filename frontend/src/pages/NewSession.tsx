import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type CharacterIn } from '../api'

// Mirror simstars.config's scope guardrails so the form can validate
// client-side before ever hitting the API - kept in sync by hand, this is
// a small local app, not generated from the backend.
const MIN_CHARACTERS = 3
const MAX_CHARACTERS = 5
const MIN_LOCATIONS = 2
const MAX_LOCATIONS = 4

function emptyCharacter(startingLocation: string): CharacterIn {
  return { name: '', role: '', traits: '', starting_location: startingLocation }
}

export function NewSession() {
  const navigate = useNavigate()
  const [worldDescription, setWorldDescription] = useState('')
  const [locations, setLocations] = useState<string[]>(['', ''])
  const [characters, setCharacters] = useState<CharacterIn[]>([
    emptyCharacter(''),
    emptyCharacter(''),
    emptyCharacter(''),
  ])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function updateLocation(index: number, value: string) {
    const next = [...locations]
    const oldValue = next[index]
    next[index] = value
    setLocations(next)
    // keep any character pointed at the old location name pointed at the new one
    setCharacters((chars) =>
      chars.map((c) => (c.starting_location === oldValue ? { ...c, starting_location: value } : c)),
    )
  }

  function addLocation() {
    if (locations.length >= MAX_LOCATIONS) return
    setLocations([...locations, ''])
  }

  function removeLocation(index: number) {
    if (locations.length <= MIN_LOCATIONS) return
    setLocations(locations.filter((_, i) => i !== index))
  }

  function updateCharacter(index: number, field: keyof CharacterIn, value: string) {
    setCharacters((chars) => chars.map((c, i) => (i === index ? { ...c, [field]: value } : c)))
  }

  function addCharacter() {
    if (characters.length >= MAX_CHARACTERS) return
    setCharacters([...characters, emptyCharacter(locations[0] ?? '')])
  }

  function removeCharacter(index: number) {
    if (characters.length <= MIN_CHARACTERS) return
    setCharacters(characters.filter((_, i) => i !== index))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      const session = await api.createSession({
        world_description: worldDescription,
        locations: locations.map((l) => l.trim()).filter(Boolean),
        characters,
      })
      navigate(`/sessions/${session.id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setSubmitting(false)
    }
  }

  return (
    <div className="page">
      <h1>New Session</h1>

      <form onSubmit={handleSubmit}>
        <label className="field">
          <span>World / setting</span>
          <textarea
            required
            value={worldDescription}
            onChange={(e) => setWorldDescription(e.target.value)}
            placeholder="e.g. A family-owned bakery, closed for the night, the evening before its 40th anniversary open house"
          />
        </label>

        <fieldset>
          <legend>
            Locations ({MIN_LOCATIONS}-{MAX_LOCATIONS})
          </legend>
          {locations.map((loc, i) => (
            <div className="row" key={i}>
              <input required value={loc} onChange={(e) => updateLocation(i, e.target.value)} placeholder="Kitchen" />
              <button type="button" onClick={() => removeLocation(i)} disabled={locations.length <= MIN_LOCATIONS}>
                Remove
              </button>
            </div>
          ))}
          <button type="button" onClick={addLocation} disabled={locations.length >= MAX_LOCATIONS}>
            + Add location
          </button>
        </fieldset>

        <fieldset>
          <legend>
            Cast ({MIN_CHARACTERS}-{MAX_CHARACTERS} characters - a sentence or two per character is enough, the rest
            gets invented)
          </legend>
          {characters.map((c, i) => (
            <div className="character-row" key={i}>
              <div className="row">
                <input
                  required
                  value={c.name}
                  onChange={(e) => updateCharacter(i, 'name', e.target.value)}
                  placeholder="Name"
                />
                <input
                  required
                  value={c.role}
                  onChange={(e) => updateCharacter(i, 'role', e.target.value)}
                  placeholder="Role"
                />
                <select
                  required
                  value={c.starting_location}
                  onChange={(e) => updateCharacter(i, 'starting_location', e.target.value)}
                >
                  <option value="" disabled>
                    Starting location
                  </option>
                  {locations.filter(Boolean).map((loc) => (
                    <option key={loc} value={loc}>
                      {loc}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => removeCharacter(i)}
                  disabled={characters.length <= MIN_CHARACTERS}
                >
                  Remove
                </button>
              </div>
              <input
                required
                className="traits-input"
                value={c.traits}
                onChange={(e) => updateCharacter(i, 'traits', e.target.value)}
                placeholder="Traits/vibe"
              />
            </div>
          ))}
          <button type="button" onClick={addCharacter} disabled={characters.length >= MAX_CHARACTERS}>
            + Add character
          </button>
        </fieldset>

        {error && <p className="error">{error}</p>}

        <button className="button" type="submit" disabled={submitting}>
          {submitting ? 'Enriching cast and world...' : 'Create Session'}
        </button>
      </form>
    </div>
  )
}
