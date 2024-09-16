import abc
import itertools
from typing import Generic, TypeVar
from collections.abc import Iterable
import attr

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
StateT = TypeVar("StateT")


@attr.s(frozen=True, kw_only=True)
class StateTransition(Generic[StateT, OutputT]):
    state: StateT | None = attr.ib(default=None)
    outputs: Iterable[OutputT] | None = attr.ib(factory=list)


class StateMachine(Generic[InputT, OutputT, StateT]):
    """A state machine has:
        A set S of states.
        A set X of input events.
        A set Y of output events.

    These sets (S, X, Y) don’t need to be finite. In an actual application, X and Y are probably going to be messages.
    Typically, X and Y are disjoint, although this does not necessarily need to be the case. If X and Y overlap—in other
    words, the output of the state machine can be fed back in as an input—the model can become a powerful computation
    engine (even Turing-complete), and you can define some wild “self-executing” state machines.
    """

    # ----------------------------------------------------------------------------------------------------------------
    # Public interface to the state machine.

    def process_startup(self, state: StateT):
        transition = self.startup(state)
        if not transition or not transition.state:
            return self._concatenate(transition, self._process_quiesce(state=state))

        return self._concatenate(transition, self._process_quiesce(state=transition.state))

    def process_input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        transition = self.input(state=state, input=input)
        if not transition or not transition.state:
            return transition

        return self._concatenate(transition, self._process_quiesce(transition.state))

    # -----------------------------------------------------------------------------------------------------------------
    # Definition of the state machine.

    @abc.abstractmethod
    def startup(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        """Implement P function.

        Function P describes how the state machine quiesces when the app starts up in initial state s0 ∈ S:
            P : S → (S ✕ Y*)
        where
            P(s) = (s’, [y1, … , yn])
        means that in state s ∈ S, the state machine transitions to state s’ ∈ S and outputs the sequence of events
        y1 ∈ Y, …, yn ∈ Y. When the Controller starts up in state s0 ∈ S, it goes through the following sequence:
            P(s0) = (s1, y1)		outputting y1 ∈ Y*
            P(s1) = (s2, y2)		outputting y2 ∈ Y*
            …
        until it reaches a “null” transition
            P(sn) = (sn, []).
        """

    @abc.abstractmethod
    def input(self, state: StateT, input: InputT) -> StateTransition[StateT, OutputT] | None:
        """Implement Q function.

        Function (Q) describes how the state machine reacts to inputs events:
            Q : S → X → (S ✕ Y*)
        where
            Q(s)(x) = (s’, [y1, … , yn])
        means that in state s ∈ S, given input event x ∈ X, the state machine transitions to state s’ ∈ S and outputs
        the sequence of events y1 ∈ Y, …, yn ∈ Y.
        """

    @abc.abstractmethod
    def quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        """Implement R function.

        Function R is similar to P but describes how the state quiesces after a transition from Q. When the Controller
        is in state s ∈ S and receives input event x ∈ X, it goes through the following sequence
            Q(s)(x) = (s0, y0)	outputting y0 ∈ Y*
            R(s0) = (s1, y1)		outputting y1 ∈ Y*
            R(s1) = (s2, y2)		outputting y2 ∈ Y*
            …
        until it reaches a “null” transition
            R(sn) = (sn, []).
        """

    # -----------------------------------------------------------------------------------------------------------------
    # Private helper functions.

    def _process_quiesce(self, state: StateT) -> StateTransition[StateT, OutputT] | None:
        transition = self.quiesce(state=state)
        if not transition or not transition.state:
            return transition
        return self._concatenate(transition, self.quiesce(state=transition.state))

    @classmethod
    def _concatenate(
        cls,
        transition_1: StateTransition[StateT, OutputT] | None,
        transition_2: StateTransition[StateT, OutputT] | None,
    ) -> StateTransition[StateT, OutputT]:
        if transition_1 and transition_2:
            return StateTransition(
                state=transition_2.state or transition_1.state,
                outputs=itertools.chain(transition_1.outputs, transition_2.outputs)
            )
        return transition_1 or transition_2
