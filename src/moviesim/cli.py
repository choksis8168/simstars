"""Typer CLI. Holds no logic beyond argument parsing/prompting and printing
- everything else goes through pipeline.py. See docs/design.md "CLI (Typer)".
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from moviesim import pipeline
from moviesim.config import MAX_CHARACTERS, MAX_LOCATIONS, MIN_CHARACTERS, MIN_LOCATIONS
from moviesim.models import Event, Screenplay

app = typer.Typer(help="Create the characters. Set the world. Press play. Watch a movie emerge.")
console = Console()


@app.command()
def new() -> None:
    """Define characters + world, interactively, and save a session."""
    console.print("[bold]World[/bold]")
    world_description = typer.prompt("Describe the world/setting")

    console.print(f"\nLocations ({MIN_LOCATIONS}-{MAX_LOCATIONS}), comma-separated")
    locations = [loc.strip() for loc in typer.prompt("Locations").split(",") if loc.strip()]

    console.print(f"\n[bold]Cast[/bold] ({MIN_CHARACTERS}-{MAX_CHARACTERS} characters)")
    specs: list[pipeline.CharacterSpec] = []
    while True:
        console.print(f"\n-- Character {len(specs) + 1} --")
        name = typer.prompt("Name")
        role = typer.prompt("Role")
        traits = typer.prompt("Traits/vibe (a sentence or two is enough - the rest gets invented)")
        console.print(f"Locations: {', '.join(locations)}")
        starting_location = typer.prompt("Starting location", default=locations[0])
        specs.append(pipeline.CharacterSpec(name, role, traits, starting_location))

        if len(specs) >= MAX_CHARACTERS:
            break
        if len(specs) >= MIN_CHARACTERS and not typer.confirm("Add another character?", default=False):
            break

    console.print("\n[dim]Enriching cast and world...[/dim]")
    session = pipeline.new_session(world_description, locations, specs)
    console.print(f"\n[bold green]Session created: {session.id}[/bold green]")
    console.print(f"Run [bold]moviesim script {session.id}[/bold] to preview the story for free, or")
    console.print(f"[bold]moviesim play {session.id}[/bold] to produce the full movie.")


def _print_screenplay(events: list[Event], screenplay: Screenplay, end_reason) -> None:
    for scene in screenplay.scenes:
        console.print(f"\n[bold]{scene.heading}[/bold]")
        if scene.sfx_cues:
            console.print(f"[dim]SFX: {', '.join(scene.sfx_cues)}[/dim]")
        if scene.music_cue:
            console.print(f"[dim]Music: {scene.music_cue}[/dim]")
        for line in scene.lines:
            console.print(line)
    console.print(f"\n[dim]End reason: {end_reason.value}[/dim]")


@app.command()
def script(session_id: str, note: str = typer.Option(None, "--note", help="Producer's note to steer this run.")) -> None:
    """Run GENERATE only (simulate + critic + screenplay) and print it. No ElevenLabs cost."""
    console.print("[dim]Simulating...[/dim]")
    run, events, screenplay = pipeline.generate(session_id, note)
    _print_screenplay(events, screenplay, run.end_reason)
    console.print(f"\n[bold]Run: {run.id}[/bold] ({run.critic_attempts} attempt(s))")


@app.command()
def play(session_id: str, note: str = typer.Option(None, "--note", help="Producer's note to steer this run.")) -> None:
    """Full generate -> critic -> screenplay -> produce -> release."""
    console.print("[dim]Simulating...[/dim]")
    run = pipeline.play(session_id, note)
    console.print(f"\n[bold green]Movie ready: {run.final_audio_path}[/bold green]")
    console.print(f"Run: {run.id}  |  End reason: {run.end_reason.value}  |  Critic attempts: {run.critic_attempts}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
