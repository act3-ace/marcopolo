# Copyright (c) 2023 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The following code is modified from uber-research/poet
# (https://github.com/uber-research/poet)
# under the Apache 2.0 License.

import datetime
import json
import logging
import os
import pickle
import time
from argparse import Namespace
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any, cast, Optional, Union

# https://matplotlib.org/2.0.2/faq/usage_faq.html#non-interactive-example
# https://github.com/matplotlib/matplotlib/issues/3466/
# https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
import matplotlib  # type: ignore[import] # pylint disable=wrong-import-position
import neat  # type: ignore[import]
import numpy as np
from numpy.random import PCG64DXSM, Generator

from marco_polo.tools.types import PathString

from marco_polo.envs._base.env_params import (
    EnvParams,
)  # pylint: disable=wrong-import-position

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import] # pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)


#######################################################
## Factories
#######################################################
def get_env_param_class(args: Namespace) -> Callable[..., Any]:
    """Returns the class to use, based on input arguments

    Parameters
    ----------
    args: argparse.Namespace
        arguments that were passed to the `main()` function

    Returns
    -------
    class
        the class to use in creating env parameter objects
    """
    return CppnEnvParams


#######################################################
## Auxiliary Class
#######################################################
class PrettyGenome(neat.DefaultGenome):  # type: ignore[misc] # DefaultGenome is Any
    """DefaultGenome extended with a human readable string representations

    This can be used as a direct replacement for neat.DefaultGeome when a more
    useful string representatinos is desired. (see __str__ method for details)

    The class is intended to be accessed via the CppnEnvParams class

    See Also
    --------
    neat.DefaultGenome for details of the class
    """

    def __str__(self) -> str:
        """Return a string representation of the genome

        An example of the output is:
        Fitness: None
        Nodes:
            0 DefaultNodeGene(key=0, bias=-0.16533354, response=4.416987, activation=identity, aggregation=sum)
            1 DefaultNodeGene(key=1, bias=-0.19784433, response=1.0, activation=sin, aggregation=sum)
        Connections:
            DefaultConnectionGene(key=(-1, 1), weight=1.03887474446, enabled=True)
            DefaultConnectionGene(key=(1, 0), weight=-0.12996966813, enabled=True)
        """
        connections = [c for c in self.connections.values() if c.enabled]
        connections.sort()
        output = "Fitness: {0}\nNodes:".format(self.fitness)
        for key, node in self.nodes.items():
            output += "\n\t{0} {1!s}".format(key, node)
        output += "\nConnections:"
        for conn in connections:
            output += "\n\t" + str(conn)
        return output


#######################################################
## Main Class
#######################################################
class CppnEnvParams(EnvParams):
    """Network that can evolve via a genetic algorithm

    CPPN = Compositional pattern-producing network (see _[1] for details)

    This class stores details of the genome and can spawn off mutated copies.
    It is essentially a wrapper around PrettyGenome that adds input, output,
    mutation, and plotting. The evolution process is done using NEAT _[2]

    Class Attributes
    ----------------
    x: np.array[float]
        The x values of the environment

    Attributes
    ----------
    parent: PrettyGenome | None
        The parent of the genome in this instance
    seed: int | Arry[int]
        random number seed
    np_random: numpy.random.Generator
        Rnadom number generator state
    cppn_config_path: str
        path to cppn config file
    genome_path: str
        path to genome file
    cppn_config: neat.Config
        Configuration for the CPPN system ????
    cppn_genome: PrettyGenome
        The actual genome
    altitude_fn: Callable
        the function to calculate the altitude

    References
    ----------
    .. [1] Kenneth O. Stanley. 2007. Compositional pattern producing networks:
           A novel abstraction of development. Genetic Programming and
           Evolvable Machines 8, 2 (Jun 2007), 131–162.
           https://doi.org/10.1007/s10710-007-9028-8

    .. [2] K. O. Stanley and R. Miikkulainen, "Efficient evolution of neural
           network topologies," Proceedings of the 2002 Congress on
           Evolutionary Computation. CEC'02 (Cat. No.02TH8600), Honolulu, HI,
           USA, 2002, pp. 1757-1762 vol.2, doi: 10.1109/CEC.2002.1004508.
    """

    # the shared X values for the environments
    x = np.array([(i - 200 / 2.0) / (200 / 2.0) for i in range(200)])
    # note, this is the same as: x = np.linspace(-1, 0.99, 200)
    # They will differ by ~10^-16 due to machine precision, so a direct output
    # comparison will have minor differences.

    def __init__(
        self,
        args: Namespace,
        seed: Optional[int] = None,
        cppn_config_path: str = "config-cppn",
        genome_path: Optional[PathString] = None,
    ) -> None:
        """Create and intialize the genome wrapper

        The object's evolution and other parameters are controlled by the
        config file. Additionally, a path to an existing genome can be
        specified. If so, it will be read in. If not, default parameters
        will be used in constructing the object.

        Parameters
        ----------
        args: Namespace
            The program parameters given on startup
        seed: int or None, default = None
            seed to use for the random number generator
        cppn_config_path: str, default = 'config-cppn'
            path to use for the cppn config. This is relative to the location
            of this file.
        genome_path: str, optional
            path to existing genome to use
        """
        # https://stackoverflow.com/questions/55231989/optional-union-in-type-hint

        super().__init__(args)
        self.parent: Union[str, None] = None
        if seed:
            self.seed = seed
        else:
            self.seed = args.master_seed
        # These parameters are not currently stateful
        #  objects are pulled from the fiber pull, but not reset in the pool
        #  so it doesn't account for stateful computation.
        # Updating calls in es.py to push the updated objects into the pool
        # handles this issue.
        # This is a mistake in the original POET code.
        self.np_random = Generator(PCG64DXSM(seed=self.seed))
        self.cppn_config_path = os.path.join(
            os.path.dirname(__file__), cppn_config_path
        )
        self.genome_path = genome_path
        self.cppn_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.cppn_config_path,
        )
        self.altitude_fn = lambda x: x
        if genome_path is not None:
            with open(genome_path, mode="rb") as file:
                self.cppn_genome = pickle.load(file)
        else:
            start_cppn_genome = PrettyGenome("0")
            start_cppn_genome.configure_new(self.cppn_config.genome_config)
            start_cppn_genome.nodes[0].activation = "identity"
            self.cppn_genome = start_cppn_genome
        self.reset_altitude_fn()

    def get_name(self) -> str:
        """Return the name of the genome"""
        # LOGGER.info(str(hash(str(self.cppn_genome))))
        # LOGGER.info(str(self.cppn_genome))
        return str(hash(str(self.cppn_genome)))

    def reset_altitude_fn(self) -> None:
        """Reset the altitude function of the genome from the network

        This would be used when a new genome is created or mutated to
        replace the default function.

        Side-effects
        ------------
        Replaces the current altitude function
        """
        net = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.cppn_config)
        self.altitude_fn = net.activate

    def get_mutated_params(self) -> "CppnEnvParams":
        """Return a mutated copy of the genome

        The parameters for the mutation are given in the config file
        used to create the genome.

        Side-effects
        ------------
        None

        Returns
        -------
        CppnEnvParams
            A new CppnEnvParam object that is a mutated copy of the original
        """
        while True:
            mutated = copy_genome(self.cppn_genome)
            mutated.nodes[0].response = 1.0
            # key is a unique identifier for this genome
            mutated.key = datetime.datetime.utcnow().isoformat()
            # this controls the actual mutation of the structure, connections,
            # and genes
            mutated.mutate(self.cppn_config.genome_config)
            distance = self.cppn_genome.distance(
                mutated, self.cppn_config.genome_config
            )
            if not (is_genome_valid(mutated) and (distance > 0)):
                continue
            net = neat.nn.FeedForwardNetwork.create(mutated, self.cppn_config)
            yvals = np.array([net.activate((xi,)) for xi in self.x])
            yvals -= yvals[0]  # normalize to start at altitude 0
            # TODO: yvals[0] is always zero, so why np.abs for the threshold_?
            threshold_ = np.abs(np.max(yvals))
            if threshold_ <= 0:  # TODO: essentially theshold_ != 0 ??
                continue
            # the significance of these numbers is not clear
            if threshold_ < 0.25:
                mutated.nodes[0].response = (
                    self.np_random.random() / 2 + 0.25
                ) / threshold_
            if threshold_ > 16:
                mutated.nodes[0].response = (
                    self.np_random.random() * 4 + 12
                ) / threshold_

            # now that the genome is created, make the wrapper
            res = CppnEnvParams(
                self.args, seed=self.np_random.integers(low=2**32 - 1, dtype=int)
            )
            res.cppn_genome = mutated
            res.reset_altitude_fn()
            res.parent = self.get_name()

            return res

    def xy(self) -> dict[str, list[float]]:
        """Return the x,y values of the environment

        Returns
        -------
        dict[str, list[float]]
            a dict of lists of the x,y values in the form:
            {'x': [x_values], 'y': [y_values]}
        """
        net = neat.nn.FeedForwardNetwork.create(self.cppn_genome, self.cppn_config)
        yvals = np.array([net.activate((xi,))[0] for xi in self.x])
        return {"x": self.x.tolist(), "y": yvals.tolist()}

    def save_xy(self, folder: PathString = "/tmp") -> None:
        """Create and save a plot of x,y values for the environment

        Parameters
        ----------
        folder: PathString, default = "/tmp"
            folder where the plot will be saved.

        Side-effects
        ------------
        Writes the plot to disk, overwriting any plot in the folder that was
        previously created by this function.
        """
        # with open(folder + '/' + self.cppn_genome.key + '_xy.json', 'w') as f:
        xy_vals = self.xy()
        x_vals, y_vals = xy_vals["x"], xy_vals["y"]
        with open(os.path.join(folder, "_xy.json"), mode="w", encoding="utf8") as file:
            file.write(json.dumps({"x": x_vals, "y": y_vals}))

        # Plot environment function
        plt.plot(x_vals, y_vals)
        plt.savefig(os.path.join(folder, "terrain.png"))
        plt.close()

    def to_json(self) -> str:
        """Return JSON representation of the object

        Returns
        -------
        str
            A JSON dump of a dict with the following keys:
            'cppn_config_path' -  path to the configuration file for the genome
            'genome_path' - path to the genome
            'parent' - the parent of this genome
            'np_random' - the random state
        """
        return json.dumps(
            {
                "cppn_config_path": self.cppn_config_path,
                "genome_path": self.genome_path,
                "parent": self.parent,
                "np_random": self.np_random.bit_generator.state,
            }
        )

    def save_genome(self, file_path: Optional[PathString] = None) -> None:
        """Save the genome (not the whole class) to a file

        Parameters
        ----------
        file_path: PathString, optional
            path to write file to. If omitted, a deafult path is selected
        """
        if file_path is None:
            file_path = "/tmp/genome_{}_saved.pickle".format(time.time())

        with open(file_path, mode="wb") as file:
            pickle.dump(self.cppn_genome, file)

    def _save(self, folder: PathString) -> None:
        """Saves the object to the given folder

        The components saved are:
        * a pickle file of the genome
          saved to _genome.pkl
        * the raw xy data
          saved to _xy.json
        * an XY plot of the data
          saved to terrain.png
        * the configuration (see self.to_json)
          saved to _config.json
        * a human readble version of the genome (see PrettyGenome.__str__)
          saved to "genome.txt"

        Parameters
        ----------
        folder: PathString
            path to store data in
        """
        os.makedirs(folder, exist_ok=True)
        self.save_genome(os.path.join(folder, "_genome.pkl"))
        self.save_xy(folder)

        with open(
            os.path.join(folder, "_config.json"), mode="w", encoding="utf8"
        ) as file:
            file.write(self.to_json())

        with open(
            os.path.join(folder, "genome.txt"), mode="w", encoding="utf8"
        ) as file:
            file.write(str(self.cppn_genome))

    def checkpoint(self, folder: PathString) -> None:
        """Save the current state of the object

        Parameters
        ----------
        folder: PathString
            path to store data in
        """
        self._save(os.path.join(folder, "cppn"))

    def reload(self, folder: PathString) -> None:
        """Load an object from disk

        Parameters
        ----------
        folder: PathString
            path to load data from

        Side-Effects
        ------------
        Overwrites this instance with the data from disk
        """
        dir_path = os.path.join(folder, "cppn")

        with open(os.path.join(dir_path, "_genome.pkl"), mode="rb") as file:
            self.cppn_genome = pickle.load(file)

        # create prng, set state within open()
        self.np_random = Generator(PCG64DXSM())

        with open(
            os.path.join(dir_path, "_config.json"), mode="r", encoding="utf8"
        ) as file:
            json_dict = json.load(file)

        self.cppn_config_path = json_dict["cppn_config_path"]
        self.genome_path = json_dict["genome_path"]
        self.parent = json_dict["parent"]
        self.np_random.bit_generator.state = json_dict["np_random"]
        self.reset_altitude_fn()


def copy_genome(genome: PrettyGenome) -> PrettyGenome:
    """Return a duplicate of the genome

    Parameters
    ----------
    genome: PrettyGenome
        the genome to copy

    Side-Effects
    ------------
    The implementation will write the state to disk

    Returns
    -------
    PrettyGenome
        a copy of the genome
    """
    # write object to disk, then read back into a new object
    file_path = "/tmp/genome_{}.pickle".format(time.time())
    with open(file_path, mode="wb") as file:
        pickle.dump(genome, file)
    with open(file_path, mode="rb") as file:
        return cast(PrettyGenome, pickle.load(file))


def is_genome_valid(genome: PrettyGenome) -> bool:
    """Return true if the genome is valid

    'valid' in this sense appears to mean that the connections graph
    makes it from the input to the output. The current config-cppn has a
    single input and output (see, num_inputs=1 and num_outputs=1). It's
    unclear if/how this would need to change if the number of inputs/outputs
    was changed in the config file.

    Parameters
    ----------
    genome: PrettyGenome
        the genome to test

    Returns
    -------
    bool
        whether the genome is valid
    """
    # graph tracks forward connections. If a connects to b, c, and d:
    # graph[a] = [b, c, d]
    graph = defaultdict(list)
    for key in genome.connections.keys():
        graph[key[0]].append(key[1])
    # this is walking along the graph, starting at the input (key -1 by
    # convention for a single input), trying to find the output (key 0 by
    # convention for a single output)
    # This code may have potential for an infinite loop with a circular
    # network i.e. b in graph[a] and a in graph[b]
    # I assume the mutation code disallows that scenario, but I haven't
    # confirmed that.
    queue = deque([-1])
    while len(queue) > 0:
        cur = queue.popleft()
        if cur == 0:  # network leads to the output, this is valid
            return True
        if cur not in graph:  # dead end in network path, ignore
            continue
        for node in graph[cur]:  # add nodes connected to this one
            # could probably add a check for zero here too, by savings would
            # be minimal for the added complexity
            queue.append(node)
    return False
