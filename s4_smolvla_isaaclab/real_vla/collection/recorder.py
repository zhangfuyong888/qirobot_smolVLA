"""Collection-side recorder: forwards policy streams to the episode writer."""

from __future__ import annotations

from real_vla.collection.episode_writer import EpisodeWriter
from real_vla.collection.schema import PolicyState, PublishedCommand


class Recorder:
    def __init__(self, writer: EpisodeWriter) -> None:
        self.writer = writer
        self.enabled = False

    def start(self) -> None:
        self.enabled = True

    def stop(self) -> None:
        self.enabled = False

    def on_state(self, state: PolicyState) -> None:
        if self.enabled:
            self.writer.record_state(state)

    def on_action(self, command: PublishedCommand) -> None:
        if self.enabled:
            self.writer.record_action(command)
