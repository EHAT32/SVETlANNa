from typing import Iterable
from .elements import Element
from . import Wavefront, SimulationParameters

import torch
from torch import nn


class RecurrentReservoir(nn.Module):
    """
    A recurrent reservoir. Supposed to by an ineducable element of a setup,
    which means that all elements mustn't have parameters tu update.
    All parameters are bounded or defined as a buffer!
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        forward_elements: Iterable[Element],
        backward_elements: Iterable[Element],
        init_hidden: Wavefront | None = None,
        device: str | torch.device = torch.get_default_device(),
    ) -> None:
        """
        Parameters
        ----------
        forward_elements : Iterable[Element]
            A set of optical elements...
        backward_elements : Iterable[Element]
            A set of optical elements for a feedback (to get a hidden state).
        """
        super().__init__()

        self.simulation_parameters = simulation_parameters

        self.__device = device
        if self.simulation_parameters.device is not self.__device:
            # TODO: right way to compare devices? check how it works?
            self.simulation_parameters.__device = self.simulation_parameters.to(self.__device)

        # TODO: check if all parameters are bounded oa a buffer?
        self.forward_elements = [el.to(self.__device) for el in forward_elements]
        self.backward_elements = [el.to(self.__device) for el in backward_elements]

        if init_hidden is None:
            self.hidden_state = self.reset_hidden
        else:
            self.hidden_state = init_hidden.to(self.__device)

    def reset_hidden(self):
        """
        Resets a "hidden" wavefront.

        Returns
        -------
        Wavefront
            A zero wavefront as an initial "hidden" wavefront.
        """
        return Wavefront(
            torch.zeros(
                # TODO: right way to get a wavefront size?
                size=self.simulation_parameters.axes_size(
                    self.simulation_parameters.axes.__dir__()
                )
            )
        ).to(self.__device)

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            An output wavefront after propagation of an input wavefront (+ hidden wavefront)
            through a forward net (output of the network).
        """
        output_wavefront = input_wavefront + self.hidden_state
        for el in self.forward_elements:
            output_wavefront = el.forward(output_wavefront)

        # get a next hidden by propagation of the output
        hidden_state = output_wavefront
        for el in self.backward_elements:
            hidden_state = el.forward(hidden_state)
        self.hidden_state = hidden_state  # update hidden state

        return output_wavefront

    def to(self, device: str | torch.device | int) -> 'RecurrentReservoir':
        if self.__device == torch.device(device):
            return self

        return RecurrentReservoir(
            simulation_parameters=self.simulation_parameters,
            forward_elements=self.forward_elements,
            backward_elements=self.backward_elements,
            init_hidden=self.hidden_state,
            device=device
        )

    @property
    def device(self) -> str | torch.device | int:
        return self.__device
