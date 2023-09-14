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


"""Wrappers used in MarcoPolo"""
import argparse
import copy
import json
import logging
import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator  # for type hinting
from typing import Any, cast, ClassVar, Optional, Union  # for type hinting

from pettingzoo.utils import agent_selector  # type: ignore[import]
from pettingzoo.utils.conversions import (  # type: ignore[import]
    aec_to_parallel_wrapper,
    parallel_to_aec_wrapper,
)
from pettingzoo.utils.wrappers import OrderEnforcingWrapper  # type: ignore[import]
from tianshou.env.pettingzoo_env import PettingZooEnv  # type: ignore[import]

from marco_polo.base_classes.model import BaseModel
from marco_polo.envs._base.env_params import EnvParams
from marco_polo.tools.iotools import NumpyDecoder, NumpyEncoder
from marco_polo.tools.types import (
    AgentId,
    EnvId,
    FloatArray,
    PathString,
    RenderFrame,
    Role,
    TeamId,
)  # for type hinting

logger = logging.getLogger(__name__)


RunTask = tuple["EnvHistory", "TeamHistory"]


################################################################################
## Environment Stats Class
################################################################################
class EnvHistoryStats:
    """Stats for an EnvHistory object

    Attributes
    ----------
    team : TeamHistory
        current team of agents for the env
    iterations_lifetime : int
        Number of iterations in enviroments entire history across epochs
    iterations_current : int
        Current iteration in epoch
    created_at : int
        Epoch created at
    recent_scores : deque(maxlen=5)
        The 5 most recent scores achieved
    transfer_threshold : float
        Score threshold for transfers to happen
    best_score : float
        Lifetime best score on this env
        default: -infinity if no run has been done
    best_team : TeamHistory
        Team of agents that scored the best score on this env
    """

    def __init__(self) -> None:
        self.team: TeamHistory = NoneTeam(None)

        self.iterations_lifetime: int = 0
        self.iterations_current: int = 0
        self.created_at: int = 1

        #  https://stackoverflow.com/q/5944708/how-to-force-a-list-to-a-fixed-size
        self.recent_scores = deque[float](maxlen=5)
        self.transfer_threshold = float("-inf")

        self.best_score = float("-inf")
        self.best_team: TeamHistory = NoneTeam(None)

    def toJSON(self) -> str:  # pylint: disable=invalid-name
        """Return a json string version of the object"""
        tmp = {k: v for k, v in self.__dict__.items() if k not in {"team", "best_team"}}

        tmp["team"] = self.team.id
        tmp["best_team"] = self.best_team.id
        tmp["recent_scores"] = list(self.recent_scores)

        return json.dumps(tmp, cls=NumpyEncoder)

    def fromJSON(
        self, data: str, team_dict: dict[str, "TeamHistory"]
    ) -> None:  # pylint: disable=invalid-name
        """Fill this object with data from the given JSON data

        In addition to reading the JSON data, teams are pulled by
        name from the given dict of teams

        Parameters
        ----------
        data : str
            JSON string with object data
        team_dict : dict[str, TeamHistory]
            dictionary of known teams

        Side-Effects
        ------------
        object data is completely replaced by data from the JSON data
        """
        for key, val in json.loads(data).items():
            if key == "team":
                self.team = team_dict[val]
            elif key == "best_team":
                self.best_team = team_dict.get(val, NoneTeam(None))
            elif key == "best_score":
                self.best_score = float(val)
            elif key == "transfer_threshold":
                self.transfer_threshold = float(val)
            elif key == "recent_scores":
                for i in val:
                    self.recent_scores.append(i)
            else:
                self.__dict__[key] = val

    def __str__(self) -> str:
        return "\n".join([f"{key}: {val}" for key, val in self.__dict__.items()])


################################################################################
## Environment History Class
################################################################################
class EnvHistory:
    """Env wrapper class

    This is a wrappper around the env_params object, with the
    information need to create an instance of the env class.
    Attributes accessed for this class that don't exist are instead
    pulled from the underlying env_params

    Parameters
    ----------
    env_param : EnvParams
        parameters to use for this environment
    env_creator : class/function
        Class or function that can be used to create the env instance

    Attributes
    ----------
    id : str
        A human readable ID of the object
        format is "Env_X" where X is the numeric index of the object
    env_param : EnvParams
        parameter object for the environment
    env_creator : class/function
        Class or function that can be used to create the env instance
    stats : EnvHistoryStats
        Statistics for this env
    """

    # this is a counter of the total number of Envs that have been
    # created (not just the number that are currently active). It
    # is used to form the id string of a new env object
    id_counter: ClassVar[int] = 0

    def __init__(self, env_param: EnvParams, env_creator: Callable[..., Any]) -> None:
        EnvHistory.id_counter += 1
        self.id = cast(
            EnvId, f"Env_{EnvHistory.id_counter}"
        )  # pylint: disable=invalid-name
        self.env_param = env_param
        self.env_creator = env_creator
        self.stats = EnvHistoryStats()

    def __getitem__(self, key: str) -> Any:
        return self.env_param[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.env_param[key] = value

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env_param, name)

    def __repr__(self) -> str:
        return f"Env {self.id}"

    def get_env_creator(self) -> Callable[..., Any]:
        """Return the callable (function/class) to create a new env"""
        return self.env_creator

    def get_mutated_env(self) -> "EnvHistory":
        """Return a mutated copy of this object

        The mutation method is completely controlled by the
        env_params object

        Returns
        -------
        EnvHistory
            New object, mutated from this object
        """

        # Generate new parameters
        new_param = self.env_param.get_mutated_params()
        # wrap in new environment
        new_env = EnvHistory(env_param=new_param, env_creator=self.env_creator)
        # return new environment
        return new_env

    def log(self) -> None:
        """Write the object to the logger"""
        logger.info(f"EnvHistory.log() Environment {self.id}: " + str(self.stats))

    def checkpoint(self, folder: PathString) -> None:
        """Save the object to the given folder"""
        folder = os.path.join(folder, "Environments", self.id)
        os.makedirs(folder, exist_ok=True)

        tmp = {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"env_param", "env_creator"}
        }
        tmp["stats"] = tmp["stats"].toJSON()

        checkpoint_path = os.path.join(folder, "EnvHistory.json")
        with open(checkpoint_path, mode="w", encoding="utf8") as file:
            json.dump(tmp, file, cls=NumpyEncoder)

        self.env_param.checkpoint(folder)

    def reload(self, folder: PathString, team_dict: dict[str, "TeamHistory"]) -> None:
        """Replace the data in this object with data from the given folder"""
        checkpoint_path = os.path.join(folder, "EnvHistory.json")
        with open(checkpoint_path, mode="r", encoding="utf8") as file:
            data = json.load(file, cls=NumpyDecoder)

        for key, val in data.items():
            self.__dict__[key] = val

        self.stats = EnvHistoryStats()
        self.stats.fromJSON(data["stats"], team_dict)
        self.env_param.reload(folder)


################################################################################
## Team Stats Class
################################################################################
class TeamHistoryStats:
    """Currently does nothing"""

    def __init__(self) -> None:
        pass

    def toJSON(self) -> str:  # pylint: disable=invalid-name
        """Return a json string version of the object - currently {}"""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def fromJSON(self, data: str) -> None:  # pylint: disable=invalid-name
        """Currently does nothing"""


################################################################################
## Team History Class
################################################################################
class TeamHistory:
    """Team wrapper class

    This is a wrappper around the team object.
    Attributes accessed for this class that don't exist are instead
    pulled from the underlying team

    Parameters
    ----------
    agent_group : Optional[dict[Role, "AgtHistory"]]
        dict mapping agents to roles

    Attributes
    ----------
    id : str
        A human readable ID of the object
        format is "Team_X" where X is the numeric index of the object
    parent :
        The parent of this team. Updated on copy
    agent_group : Optional[dict[Role, "AgtHistory"]]
        Agents in this team
    stats : TeamHistoryStats
        Statistics for this team
    """

    # this is a counter of the total number of teams that have been
    # created (not just the number that are currently active). It
    # is used to form the id string of a new team object
    id_counter: ClassVar[int] = 0

    def __init__(self, agent_group: Optional[dict[Role, "AgtHistory"]] = None) -> None:
        TeamHistory.id_counter += 1
        self.id = cast(
            TeamId, f"Team_{TeamHistory.id_counter}"
        )  # pylint: disable=invalid-name
        self.parent: Union[str, None] = None

        if agent_group is not None:
            self.agent_group = agent_group
        self.stats = TeamHistoryStats()

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.agent_group, name)

    def __getitem__(self, key: Role) -> "AgtHistory":
        return self.agent_group[key]

    def __setitem__(self, key: Role, value: "AgtHistory") -> None:
        self.agent_group[key] = value

    def __repr__(self) -> str:
        id_info = {
            ", ".join(
                f"{role}: {str(agent)}" for role, agent in self.agent_group.items()
            )
        }
        return f"Team {self.id} - {id_info}"

    def roles(self) -> Iterable[Role]:
        """Return the roles in the team"""
        return self.agent_group.keys()

    def agents(self) -> Iterable["AgtHistory"]:
        """Return the roles in the team"""
        return self.agent_group.values()

    def items(self) -> Iterable[tuple[Role, "AgtHistory"]]:
        """Return the roles/agent pairs in the team"""
        return self.agent_group.items()

    def copy(self) -> "TeamHistory":
        """return a copy of the object"""
        agent_group_copy = {
            role: agent.copy() for role, agent in self.agent_group.items()
        }
        new_team = TeamHistory(agent_group_copy)
        new_team.stats = copy.deepcopy(self.stats)
        new_team.parent = self.id
        return new_team

    def checkpoint(self, folder: PathString) -> None:
        """Save the object to the Teams subdirectory of the given folder"""
        folder = os.path.join(folder, "Teams")
        os.makedirs(folder, exist_ok=True)

        tmp = {k: v for k, v in self.__dict__.items() if k != "agent_group"}
        tmp["stats"] = tmp["stats"].toJSON()
        tmp["team"] = {role: agent.id for role, agent in self.agent_group.items()}

        checkpoint_path = os.path.join(folder, f"{self.id}.json")
        with open(checkpoint_path, mode="w", encoding="utf8") as file:
            json.dump(tmp, file, cls=NumpyEncoder)

    def reload(
        self,
        team_file: PathString,
        agent_dict: dict[AgentId, "AgtHistory"],
    ) -> None:
        """Replace the data in this object with data from the given file"""
        with open(team_file, mode="r", encoding="utf8") as file:
            data = json.load(file, cls=NumpyDecoder)

        for key, val in data.items():
            self.__dict__[key] = val

        self.stats = TeamHistoryStats()
        self.stats.fromJSON(data["stats"])
        self.agent_group = {k: agent_dict[v] for k, v in data["team"].items()}


class NoneTeam(TeamHistory):
    """The team version of None

    This can be used in place of None when assigning a team.

    This isn't a great solution, but it simplifies some typing and
    checkpoint/reload operations

    Attributes
    ----------
    id : None
        the id is None instead of a string
    """

    def __init__(
        self, agent_group: Optional[dict[str, "AgtHistory"]] = None
    ) -> None:  # pylint: disable=super-init-not-called
        # Do NOT call super init(). It will mess up the team count
        # appease mypy with the change in type of id
        self.id = cast(TeamId, None)  # pylint: disable=invalid-name
        self.agent_group = {}

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        raise TypeError("Trying to get attribute from NoneTeam")

    def __repr__(self) -> str:
        return f"Team {self.id} - "

    def copy(self) -> "NoneTeam":
        """return a new NoneTeam (they are all the same)"""
        return NoneTeam(agent_group=None)

    def checkpoint(self, folder: PathString) -> None:
        """Save the object to the Teams subdirectory of the given folder"""
        raise TypeError("Trying to checkpoint NoneTeam")

    def reload(
        self,
        team_file: PathString,
        agent_dict: dict[AgentId, "AgtHistory"],
    ) -> None:
        """Replace the data in this object with data from the given file"""
        raise TypeError("Trying to reload from NoneTeam")


################################################################################
## Agent Stats Class
################################################################################
class AgtHistoryStats:
    """Stats for an AgtHistory object - currently does nothing"""

    def __init__(self) -> None:
        pass

    def toJSON(self) -> str:  # pylint: disable=invalid-name
        """Return a json string version of the object - currently {}"""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    def fromJSON(self, data: str) -> None:  # pylint: disable=invalid-name
        """currently does nothing"""


################################################################################
## Agent History Class
################################################################################
class AgtHistory:
    """Agent wrapper class

    This is a wrappper around the agent object.
    Attributes accessed for this class that don't exist are instead
    pulled from the underlying agent.

    Parameters
    ----------
    model : BaseModel
        The model defining this agent

    Attributes
    ----------
    id : str
        A human readable ID of the object
        format is "Agent_X" where X is the numeric index of the object
    parent : Optional[str]
        The id of this agent's parent
    model : BaseModel
        underlying model object
    stats : AgentHistoryStats
        Statistics for this agent
    """

    # this is a counter of the total number of agents that have been
    # created (not just the number that are currently active). It
    # is used to form the id string of a new agent object
    id_counter: ClassVar[int] = 0

    def __init__(self, model: BaseModel) -> None:
        AgtHistory.id_counter += 1
        self.id = cast(
            AgentId, f"Agent_{AgtHistory.id_counter}"
        )  # pylint: disable=invalid-name
        self.parent: Union[str, None] = None

        self.model = model
        self.stats = AgtHistoryStats()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.model, name)

    def __repr__(self) -> str:
        return f"Agent {self.id}"

    def copy(self) -> "AgtHistory":
        """return a copy of the object"""
        new_agent = AgtHistory(copy.deepcopy(self.model))
        new_agent.stats = copy.deepcopy(self.stats)
        new_agent.parent = self.id
        return new_agent

    def checkpoint(self, folder: PathString) -> None:
        """Save the object to the Agents subdirectory of the given folder"""
        folder = os.path.join(folder, "Agents", f"{self.id}")
        os.makedirs(folder, exist_ok=True)

        tmp = {k: v for k, v in self.__dict__.items() if k != "model"}
        tmp["stats"] = tmp["stats"].toJSON()

        checkpoint_path = os.path.join(folder, "AgentHistory.json")
        with open(checkpoint_path, mode="w", encoding="utf8") as file:
            json.dump(tmp, file, cls=NumpyEncoder)

        self.model.checkpoint(os.path.join(folder, "Model.json"))

    def reload(self, folder: PathString) -> None:
        """Replace the data in this object with data from the given file"""
        checkpoint_path = os.path.join(folder, "AgentHistory.json")
        with open(checkpoint_path, mode="r", encoding="utf8") as file:
            data = json.load(file, cls=NumpyDecoder)

        for key, val in data.items():
            self.__dict__[key] = val

        self.stats = AgtHistoryStats()
        self.stats.fromJSON(data["stats"])

        self.model.reload(os.path.join(folder, "Model.json"))


################################################################################
## No Augment Wrapper - For using vanilla gym envs
################################################################################
class NoAugmentEnv:
    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def augment(self, env_params: EnvParams) -> None:
        """Apply env parameters to the env (does nothing for this class)"""

    def seed(self, seed: int) -> None:
        """Seed the env (does nothing for this class)"""


################################################################################
## Convert Wrappers
################################################################################
class SingleAgentGame:
    """
    Game wrapper

    This sets up a wrapper to combine multiple agents to an input format for a game.
    It also reshapes returns from the game for multiple agents.

    Attributes
    ----------
    env : Gym environment
        Gym environemnt to wrap.
    rn : str
        Role-name to give agent

    Methods
    -------
    step(dict[role, action])
        Pass agent actions to the environment, get step return
    reset()
        Reset the env, return initial obs for each role
    """

    def __init__(self, env: Any, rolename: Role = Role("role1")) -> None:
        """
        Init Function

        Parameters
        ----------
        env : Gym environment
        rolename : str
            role for each agent?

        Side-Effects
        ------------
        None

        Returns
        -------
        None
        """

        # self.env = env
        # if hasattr(env, "possible_agents"):
        #     if len(env.possible_agents) > 1:
        #         logger.info("Trying to apply wrapper to multiagent game!")
        #     else:
        #         self.rolename = env.possible_agents[0]
        # else:
        #     self.rolename = "default"
        #     self.possible_agents = ["default"]

        self.env = env
        self.rolename = rolename
        self.possible_agents = [self.rolename]
        self.agents = [self.rolename]
        self.observation_space = {self.rolename: env.observation_space}
        self.action_space = {self.rolename: env.action_space}
        if not hasattr(env, "render_mode"):
            self.render_mode = "rgb_array"

    def __getattr__(self, name: str) -> Any:
        """
        Returns an attribute with ``name``, unless ``name`` starts with an underscore.

        Parameters
        ----------
        name : str
            action name to peform

        Side-Effects
        ------------
        Possibly
            Environment performs action

        Returns
        -------
        Possibly
            Environment handles returns
        """
        # check if protected
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        # pass down to environment to handle
        return getattr(self.env, name)

    def step(self, actions: dict[str, FloatArray]) -> Iterator[dict[str, Any]]:
        """
        Take one step of environment.

        The idea of this function is to get the action of every agent, pass it to the
        environment, then reshape the return to match observations and rewards to
        the appropriate agent.

        This function currently only handles 1 agent, though with looping could be
        setup to handle multiple.

        Parameters
        ----------
        actions : dict[role, actVec]
            Dictionary of actions for each agent

        Side-Effects
        ------------
        Steps the environment

        Returns
        -------
        list[dict[role, obsVec], dict[role, rewardVal], bool, "info"]
            Returns a list of observations and rewards for every agent
        """
        # get "all" actions
        action = list(actions.values())[0]

        # pass action to environment, get returns
        returns = self.env.step(action)

        # reshape for each agent and return
        return ({self.rolename: r} for r in returns)

    def reset(self) -> dict[str, Any]:
        """
        Reset the Environment

        Returns the appropriate observation vector

        Parameters
        ----------
        None

        Side-Effects
        ------------
        Resets the environment

        Returns
        -------
        dict[role, obsVec]
            Returns an observation vector for each role from the environment
        """
        # reset env, get default observations
        returns = self.env.reset()
        # reshape for each agent and return
        return {self.rolename: returns}


class Gym_to_Gymnasium:
    """Wrapper to convert Gym env to Gymnasium"""

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name == "observation_spaces":
            return self.env.observation_space

        if name == "action_spaces":
            return self.env.action_space

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Perform step and return gymnasium format results"""
        obs, reward, done, info = self.env.step(*args, **kwargs)
        return (obs, reward, done, done, info)

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Perform reset and return gymnasium format results"""
        obs = self.env.reset()
        return obs, {}

    def observation_space(self, agent: Role) -> FloatArray:
        """Return the observation space for the given agent"""
        return self.env.observation_space[agent]  # type: ignore  # from base class

    def action_space(self, agent: Role) -> FloatArray:
        """Return the action space for the given agent"""
        return self.env.action_space[agent]  # type: ignore  # from base class


class Gymnasium_To_GymEnv:
    """
    Wrapper to convert Gymnasium env's to gym envs. Legacy.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name == "observation_space":
            return self.env.observation_spaces

        if name == "action_space":
            return self.env.action_spaces

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def step(
        self, *args: Any, **kwargs: Any
    ) -> tuple[FloatArray, float, bool, dict[str, Any]]:
        obs, reward, term, _, info = self.env.step(*args, **kwargs)
        return (obs, reward, term, info)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        obs, _ = self.env.reset()
        return obs


################################################################################
## Tianshou Wrappers
################################################################################
## Taken from Petting Zoo: https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/agent_selector.py
class MP_AEC_to_Parallel(PettingZooEnv):  # type: ignore[misc] # PettingZooEnv is Any
    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        return getattr(self.env, name)

    def _observe(self) -> dict[Role, Any]:
        obs = dict()
        for agt in self.env.agents:
            obs[agt] = self.env.observe(agt)["observation"]

        return obs

    def reset(self, close: bool = False) -> tuple[Any, Any]:
        r = self.env.reset()

        if not r:
            obs = self._observe()

        self.term = {a: False for a in self.env.agents}
        self.trunc = {a: False for a in self.env.agents}

        return (obs, None)

    def _last(self, agent: Role) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        observation = self.env.observe(agent)["observation"]
        return (
            observation,
            self.env._cumulative_rewards[agent],
            self.env.terminations[agent],
            self.env.truncations[agent],
            self.env.infos[agent],
        )

    def step(
        self, action: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        # agt = next(self.a_itter)
        agt = self.env.agent_selection
        # print("Curent Agent:?", agt)
        # print(action)
        self.env.step(action[agt])
        obss = dict()
        rewards = dict()
        terms = dict()
        truncs = dict()
        infos = dict()

        for a in self.env.agents:
            obs, reward, term, trunc, info = self._last(a)
            obss[a] = obs
            rewards[a] = reward
            terms[a] = term
            truncs[a] = trunc
            infos[a] = info

        return obss, rewards, terms, truncs, infos

    def render(self, close: bool = False) -> Union[None, RenderFrame]:
        if close:
            return None
        else:
            # since env isn't defined, we can't tell what this returns
            return self.env.render()  # type: ignore[no-any-return]


class MP_PettingZooEnv(PettingZooEnv):  # type: ignore[misc] # PettingZooEnv is Any
    def __init__(
        self, setup_args: argparse.Namespace, *args: Any, **kwargs: Any
    ) -> None:
        self.setup_args = setup_args

        self.check_max_steps_per_ep = False
        if "max_steps_per_ep" in setup_args.tianshou.keys():
            self.check_max_steps_per_ep = True
            self.max_steps_per_ep = setup_args.tianshou["max_steps_per_ep"]

        super().__init__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def augment(self, env_params: EnvParams) -> None:
        """Apply env parameters to the env"""
        self.env.augment(env_params)

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the env"""
        self.env.seed(seed)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Do step, returning output in petting zoo format"""
        result = super().step(*args, **kwargs)
        self.steps += 1
        if self.check_max_steps_per_ep:
            if self.steps == self.max_steps_per_ep:
                result = (result[0], result[1], result[2], True, result[4])

        return result

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.steps = 0
        r = super().reset(*args, **kwargs)
        self.env.seed(0)
        return r[0], r[1]


def MP_parallel_to_aec(par_env: Any) -> aec_to_parallel_wrapper:
    if isinstance(par_env, aec_to_parallel_wrapper):
        return par_env.aec_env
    aec_env = MP_parallel_to_aec_wrapper(par_env)
    ordered_env = MP_OrderEnforcingWrapper(aec_env)
    return ordered_env


class MP_parallel_to_aec_wrapper(parallel_to_aec_wrapper):  # type: ignore[misc] # parallel_to_aec_wrapper is Any
    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def augment(self, env_params: EnvParams) -> None:
        self.env.augment(env_params)

    def seed(self, seed: int) -> None:
        self.env.seed(seed)

    # If the system is failing beceause it can't convert a list to an action, try
    # uncommenting this code.
    def step(self, action: FloatArray) -> None:
        # I don't know why this is here...
        # if action is not None:
        #     action = int(action)
        if (
            self.terminations[self.agent_selection]  # type: ignore  # types from base class
            or self.truncations[self.agent_selection]  # type: ignore  # types from base class
        ):
            del self._actions[self.agent_selection]  # type: ignore  # types from base class
            # TODO: this may be wrong
            assert action is not None
            self._was_dead_step(action)
            return

        self._actions[self.agent_selection] = action

        if self._agent_selector.is_last():
            obss, rews, terminations, truncations, infos = self.env.step(self._actions)

            self._observations = copy.copy(obss)
            self.terminations = copy.copy(terminations)
            self.truncations = copy.copy(truncations)
            self.infos = copy.copy(infos)
            self.rewards = copy.copy(rews)
            self._cumulative_rewards = copy.copy(rews)

            env_agent_set = set(self.env.agents)

            self.agents = self.env.agents + [
                agent
                for agent in sorted(self._observations.keys())
                if agent not in env_agent_set
            ]

            if len(self.env.agents):
                self._agent_selector = agent_selector(self.env.agents)
                self.agent_selection = self._agent_selector.reset()

            self._deads_step_first()
        else:
            if self._agent_selector.is_first():
                self._clear_rewards()

            self.agent_selection = self._agent_selector.next()


class MP_OrderEnforcingWrapper(OrderEnforcingWrapper):  # type: ignore[misc] # OrderEnforcingWrapper is Any
    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        if name in self.__dict__:
            return self.__dict__[name]

        return getattr(self.env, name)

    def augment(self, env_params: EnvParams) -> None:
        """Apply env parameters to the env"""
        self.env.augment(env_params)

    def seed(self, seed: int) -> None:
        """Seed the env"""
        self.env.seed(seed)
