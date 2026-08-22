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

  // The single source of truth for "what locations actually exist" -
  // derived, not stored, so the dropdown (which assigns starting_location)
  // and the final submission always agree byte-for-byte. `locations` state
  // itself stays untrimmed so the text inputs don't fight the user while
  // they're mid-typing (e.g. typing a name that legitimately has an
  // internal space).
  const cleanLocations = locations.map((l) => l.trim()).filter(Boolean)

  function updateLocation(index: number, value: string) {
    // Compare/reassign using trimmed values - the dropdown below always
    // offers trimmed location names (see cleanLocations), so a character's
    // starting_location is always a trimmed string; comparing against the
    // raw (possibly whitespace-padded) previous input here would silently
    // fail to match and leave that character pointed at a location name
    // that no longer exists once the untrimmed input is cleaned up on submit.
    const oldTrimmed = locations[index].trim()
    const newTrimmed = value.trim()
    const next = [...locations]
    next[index] = value
    setLocations(next)
    if (newTrimmed && oldTrimmed !== newTrimmed) {
      setCharacters((chars) =>
        chars.map((c) => (c.starting_location === oldTrimmed ? { ...c, starting_location: newTrimmed } : c)),
      )
    }
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
    setCharacters([...characters, emptyCharacter(cleanLocations[0] ?? '')])
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
        world_description: worldDescription.trim(),
        locations: cleanLocations,
        characters: characters.map((c) => ({
          name: c.name.trim(),
          role: c.role.trim(),
          traits: c.traits.trim(),
          starting_location: c.starting_location.trim(),
        })),
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
          <p className="hint">
            Role = their social/functional identity ("head cheerleader," "new student"). Traits = personality/vibe -
            that's where "good girl," "bad boy," etc. belong.
          </p>
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
                  placeholder="Role (e.g. head cheerleader, new student)"
                />
                <select
                  required
                  value={c.starting_location}
                  onChange={(e) => updateCharacter(i, 'starting_location', e.target.value)}
                >
                  <option value="" disabled>
                    Starting location
                  </option>
                  {cleanLocations.map((loc) => (
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
