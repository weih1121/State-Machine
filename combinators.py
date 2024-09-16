from typing import Generic
from collections.abc import Iterable, Callable

from state_machine import StateT, InputT, OutputT, StateMachine, StateTransition


class UnionMachine(Generic[StateT, InputT, OutputT], StateMachine[StateT, InputT, OutputT]):
    """A machine whose set of transitions is the union of an ordered sequence of other machines.

    If more than one machine results in a transition in a given state, the first one takes precedence.
    """
    def __init__(self, machines: Iterable[StateMachine[StateT, InputT, OutputT]]) -> None:
        self._machines = machines

    def startup(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        """Pick the first machine that results in a transition or None if none of them do."""
        for machine in self._machines:
            transition = machine.startup(state=state)
            if transition:
                return transition
        return None

    def input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        """Pick the first machine that results in a transition or None if node of them do."""
        for machine in self._machines:
            transition = machine.input(state=state, input=input)
            if transition:
                return transition
        return None

    def quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        """Pick the first machine that results in a transition or None if none of them do."""
        for machine in self._machines:
            transition = machine.quiesce(state=state)
            if transition:
                return transition
        return None


class SupressInputMachine(Generic[StateT, InputT, OutputT], StateMachine[StateT, InputT, OutputT]):
    """Act like other machine but ignore all inputs satisfying a predicate."""

    def __int__(self, machine: StateMachine[StateT, InputT, OutputT], ignore_input: Callable[[InputT], bool]) -> None:
        self._machine = machine
        self._ignore_input = ignore_input

    def startup(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        return self._machine.startup(state=state)

    def input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        if self._ignore_input(input):
            return None
        return self._machine.input(state=state, input=input)

    def quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        return self._machine.quiesce(state=state)


class InitialInputMachine(Generic[StateT, InputT, OutputT], StateMachine[StateT, InputT, OutputT]):
    """Send an initial input to a machine and otherwise act like that machine."""

    def __init__(self, machine: StateMachine[StateT, InputT, OutputT], initial_input: InputT) -> None:
        self._machine = machine
        self._initial_input = initial_input
        self._initialized = False

    def startup(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        return self._concatenate(self._initialize(state=state), self._machine.startup(state=state))

    def input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        return self._concatenate(self._initialize(state=state), self._machine.input(state=state, input=input))

    def quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        return self._concatenate(self._initialize(state=state), self._machine.quiesce(state=state))

    def _initialize(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        if not self._initialized:
            self._initialized = True
            return self._machine.process_input(state=state, input=self._initial_input)
        return None


class StoppingMachine(Generic[StateT, InputT, OutputT], StateMachine[StateT, InputT, OutputT]):
    """Act like some other machine until reaching a final state, and then do nothing."""

    def __init__(self, machine: StateMachine[StateT, InputT, OutputT], until: Callable[[StateT], bool]) -> None:
        self._machine = machine
        self._until = until

    def startup(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        if self._until(state):
            return None
        return self._machine.startup(state=state)

    def input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        if self._until(state):
            return None
        return self._machine.input(state=state, input=input)

    def quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        if self._until(state):
            return None
        return self._machine.quiesce(state=state)


