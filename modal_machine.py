"""A general way of adding "modes" to smaller state machines in order to build a composite "modal" state machine.

Let's say the "underlying" state machine has:
  * states of type `StateT`
  * inputs of type `InputT`
  * outputs of type `OutputT`

Given a set of modes (type variable `ModeT`), typically a small finite set, the "modal" state machine has
  * states of type `(StateT, ModeT, ModeT)`
  * inputs of type `InputT` | `ModeT`
  * outputs of type `OutputT` | `ModeT`

The three components of the modal state are:
  * the underlying state
  * the current mode
  * the target mode

For example, if the set of modes is {RUNNING, STOPPED}:
  * `(state, RUNNING, RUNNING)`: the machine is `RUNNING` in `state`.
  * `(state, RUNNING, STOPPED)`: the machine is `RUNNING` in `state` and is transitioning to `STOPPING`.
  * `(state, STOPPED, STOPPED)`: the machine is `STOPPED` in `state`.
  * `(state, STOPPED, RUNNING)`: the machine is `STOPPED` in `state` and is transitioning to `RUNNING`.

A modal input is one of:
  * an underlying input
  * a directive to set the target mode

A modal output is one of:
  * an underlying output
  * an event that the machine has transitioned to the target mode

To build a model machine, you need to provide a `ModeMachines[StateT, ModeT, InputT, OutputT]` object which provides:
  * A method `mode_machine(current_mode, target_mode)` returning the underlying state machine
    (`StateMachine[StateT, InputT, OutputT]`) defining the station behavior for this mode combination.
  * A method `can_transition_to_target_mode(state, current_mode, target_mode)` where `current_mode` != `target_mode`
    reporting whether it is permitted to transition from `current_mode` to `target_mode` in `state` and, if so,
    optionally specifying a new `target_mode`.

The first part of the `ModeMachines` allows defining different business logic for each of the M^2 combinations for a set
of M modes. In practice, you typically don't need to write N^2 different state machines. For example, it's often
appropriate to share the same underlying state machine regardless of the target mode:
  * `(state, RUNNING, _)` uses the "running" state machine.
  * `(state, STOPPED, _)` uses the "stopped" state machine.

To understand the second part of the `ModeMachines`, consider a scenario in which when there is a request to transition
to `STOPPED` mode, you want to finish the pick in progress before actually transitioning into `STOPPED`. Then you would
arrange `can_transition_to_target_mode(state, RUNNING, STOPPED)` to return `None` if `state` represents a state in
which there is a pick in progress.
"""

import abc
from typing import Generic, TypeVar

from state_machine import StateT, InputT, OutputT, StateMachine, StateTransition
import attr

ModeT = TypeVar("ModeT")


@attr.s(frozen=True, kw_only=True)
class ModalState(Generic[StateT, ModeT]):
    state: StateT = attr.ib()
    current_mode: ModeT = attr.ib()
    target_mode: ModeT = attr.ib()

    def _test_mode(self) -> bool:
        if isinstance(self.state, ModalState):
            return self.state._test_mode()
        if hasattr(self.state, "_test_mode"):
            return self.state._test_mode # Type: ignore
        return False

    def transition_to_target_mode(self, next_target_mode: ModeT | None) -> "ModalState[StateT, ModeT] | None":
        current_mode = self.target_mode
        target_mode = next_target_mode or self.target_mode
        if self.current_mode == current_mode and self.target_mode == target_mode:
            return None
        if self._test_mode():
            print(f"{current_mode=} {target_mode=}.")
        return attr.evolve(self, current_mode=current_mode, target_mode=target_mode)

    def with_target_mode(self, target_mode: ModeT) -> "ModalState[StateT, ModeT] | None":
        if self.target_mode == target_mode:
            return None
        if self._test_mode():
            print(f"{target_mode=}.")
        return attr.evolve(self, target_mode=target_mode)


class ModalInput(Generic[InputT, ModeT]):
    pass


@attr.s(frozen=True, kw_only=True)
class Input(Generic[InputT, ModeT], ModalInput[InputT, ModeT]):
    input: InputT = attr.ib()


@attr.s(fronzen=True, kw_only=True)
class ModeInput(Generic[InputT, ModeT], ModalInput[InputT, ModeT]):
    target_mode: ModeT = attr.ib()


class ModalOutput(Generic[OutputT, ModeT]):
    current_mode: ModeT = attr.ib()
    target_mode: ModeT = attr.ib()


@attr.s(frozen=True, kw_only=True)
class Output(Generic[OutputT, ModeT], ModalOutput[OutputT, ModeT]):
    output: OutputT = attr.ib()


@attr.s(frozen=True, kw_only=True)
class ModeOutput(Generic[InputT, ModeT], ModalOutput[OutputT, ModeT]):
    current_mode: ModeT = attr.ib()
    target_mode: ModeT = attr.ib()


@attr.s(frozen=True, kw_only=True)
class TransitionAllowed(Generic[ModeT]):
    next_target_mode: ModeT | None = attr.ib(default=None)


class ModeMachines(Generic[StateT, ModeT, InputT, OutputT], abc.ABC):
    @abc.abstractmethod
    def mode_machine(self, current_mode: ModeT, target_mode: ModeT) -> StateMachine[StateT, InputT, OutputT] | None:
        """Return the state machine for (current_mode, target_mode) or `None` if the mode change is invalid."""

    @abc.abstractmethod
    def can_transition_to_target_mode(self, state: ModalState[StateT, ModeT]) -> TransitionAllowed[ModeT] | None:
        """Report whether a transition from `state.current_mode` to `state.target_mode` is allowed in `state`.

        This method will only be called for states in which `current_mode != target_mode`. The possible return values:
          * `None`: The transition is not allowed.
          * `TransitionAllowed()`: Allowed transition to `(current_mode=target_mode, target_mode=target_mode)`.
          * `TransitionAllowed(next_target_mode=t)`: Allowed transition to `(current_mode=target_mode, target_mode=t)`.
        """


class ModalMachine(
    Generic[StateT, ModeT, InputT, OutputT],
    StateMachine[ModalState[StateT, ModeT], ModalInput[InputT, ModeT], ModalOutput[OutputT, ModeT]],
):
    def __init__(self, mode_machines: ModeMachines[StateT, ModeT, InputT, OutputT]) -> None:
        self._mode_machines = mode_machines

    def startup(self, state: ModalState[StateT, ModeT]) -> StateTransition[StateT, OutputT] | None:
        transition = self._transition_to_target_mode(state=state)
        if not transition:
            return None
        machine = self._mode_machine(state=state)
        if machine is None:
            return None
        return self._modal(state=state, transition=machine.startup(state=state.state))

    def input(
            self, state: ModalState[StateT, ModeT], input: ModalInput[InputT, ModeT],
    ) -> StateTransition[ModalState[StateT, ModeT], ModalOutput[OutputT, ModeT]] | None:
        match input:
            case ModeInput(target_mode=m):
                if self._is_invalid_mode_transition(state.current_mode, m):
                    print(f"Invalid mode transition, {state.current_mode} -> {m}")
                    return None
                next_state = state.with_target_mode(target_mode=m)
                if next_state is None:
                    return None
                return StateTransition[ModalState[StateT, ModeT], ModalOutput[OutputT, StateT]](
                    state=state.state,
                    outputs=[
                        ModeOutput(
                            current_mode=next_state.current_mode,
                            target_mode=next_state.target_mode,
                        )
                    ],
                )
            case Input(input=x):
                machine = self._mode_machine(state=state)
                if machine is None:
                    return None
                return self._modal(state=state, transition=machine.input(state=state.state, input=x))
        raise TypeError(f"Unexpected input {input}.")

    def quiesce(self, state: ModalState[StateT, ModeT]) -> StateTransition[StateT, OutputT] | None:
        transition = self._transition_to_target_mode(state=state)
        if transition is None:
            return None
        machine = self._mode_machine(state=state)
        if machine is None:
            return None
        return self._modal(state=state, transition=machine.quiesce(state=state.state))

    def _is_invalid_mode_transition(self, current_mode: ModeT, target_mode: ModeT) -> bool:
        # determines whether the mode transition is valid by whether a corresponding mode machine is defined
        return self._mode_machines.mode_machine(current_mode=current_mode, target_mode=target_mode) is not None

    def _transition_to_target_mode(
            self, state: ModalState[StateT, ModeT],
    ) -> StateTransition[ModalState[StateT, ModeT], ModalOutput[StateT, ModeT]] | None:
        if state.current_mode == state.target_mode:
            return None
        transition_allowed = self._mode_machines.can_transition_to_target_mode(state=state)
        if not transition_allowed:
            return None
        next_state = state.transition_to_target_mode(next_target_mode=transition_allowed.next_target_mode)
        if next_state is None:
            return None
        return StateTransition[ModalState[StateT, ModeT], ModalOutput[OutputT, ModeT]](
            state=next_state,
            outputs=[
                ModeOutput(target_mode=next_state.target_mode, current_mode=next_state.current_mode)
            ],
        )

    def _mode_machine(self, state: ModalState[StateT, ModeT]) -> StateMachine[StateT, InputT, OutputT] | None:
        return self._mode_machines.mode_machine(current_mode=state.current_mode, target_mode=state.target_mode)

    @classmethod
    def _modal(
        cls, state: ModalState[StateT, ModeT], transition: StateTransition[StateT, OutputT] | None
    ) -> StateTransition[ModalState[StateT, ModeT], ModalOutput[OutputT, ModeT]] | None:
        if not transition:
            return None
        outputs = map(lambda output: Output[OutputT, ModeT](output=output), transition.outputs)
        if transition.state is None:
            return StateTransition[ModalState[StateT, ModeT], ModalOutput[OutputT, ModeT]](outputs=outputs)
        return StateTransition[ModalState[ModalState, ModeT], ModalOutput[OutputT, ModeT]](
            state=ModalState[StateT, ModeT](
                state=state.state, current_mode=state.current_mode, target_mode=state.target_mode,
            ),
            outputs=outputs,
        )
