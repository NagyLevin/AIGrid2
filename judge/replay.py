import json
import dataclasses
import numpy as np
import typing
import types
import collections

from typing import Optional, Any

@dataclasses.dataclass(frozen=True, slots=True)
class EnvInfo:
    track: list[list[int]]
    num_players: int
    player_names: Optional[list[str]] = None

@dataclasses.dataclass(frozen=True, slots=True)
class PlayerState:
    x: int
    y: int
    vel_x: int
    vel_y: int

@dataclasses.dataclass(frozen=True, slots=True)
class State:
    turn: int
    players: list[PlayerState]

@dataclasses.dataclass(frozen=True, slots=True)
class PlayerStep:
    player_ind: int
    success: bool
    status: str = ''
    dx: Optional[int] = None
    dy: Optional[int] = None

    def __post_init__(self):
        if self.success:
            assert self.dx is not None, \
                    'dx should be defined for a successful step'
            assert self.dy is not None, \
                    'dy should be defined for a successful step'
        else:
            assert self.status, 'Failure message should be specified'

@dataclasses.dataclass(frozen=True, slots=True)
class Replay:
    """
    ``states`` contain the states before and after each step, ``steps`` contain
    the plys or steps the players made. So ``len(states) = len(steps) + 1``.
    """
    env_info: EnvInfo
    states: list[State] = dataclasses.field(default_factory=list)
    steps: list[PlayerStep] = dataclasses.field(default_factory=list)
    version: int = 1

class Encoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, (EnvInfo, PlayerState, State, PlayerStep, Replay)):
            return {
                field.name: getattr(o, field.name)
                for field in dataclasses.fields(o)
            }
        elif isinstance(o, np.integer):
            return int(o)
        else:
            return json.JSONEncoder.default(self, o)

def serialise(replay: Replay, output) -> None:
    if isinstance(output, str):
        with open(output, 'w') as f:
            json.dump(replay, f, cls=Encoder)
    else:
        json.dump(replay, output, cls=Encoder)

T = typing.TypeVar('T')
Dataclass = typing.TypeVar('Dataclass')  # specialised, dataclass type

def _construct_dataclass(target_cls: type[T], obj: dict[str] | Any, *,
                         allow_extra_keys: bool) -> T:
    # Note: this (whole function) only works if the dataclasses are not
    # subclassed
    # Also not supported: tuples and dicts (use dataclasses)
    generic_cls = typing.get_origin(target_cls)
    assert generic_cls not in [dict, tuple], f'{generic_cls} is not supported'
    if generic_cls == list:
        return target_cls(
            _construct_dataclass(
                typing.get_args(target_cls)[0],
                elem,
                allow_extra_keys=allow_extra_keys) for elem in obj)
    if (generic_cls == typing.Optional or generic_cls == typing.Union
            or generic_cls == types.UnionType):
        # Optional, Union or `|` (the last one is `UnionType`)
        # this check may not work in later than Python 3.12
        concrete_clss = typing.get_args(target_cls)
        if isinstance(obj, collections.abc.Mapping):
            # target cls should be a dataclass
            concrete_cls = [
                c for c in concrete_clss if dataclasses.is_dataclass(c)
                and _check_dataclass_compatibility(c, obj)
            ]
            if len(concrete_cls) == 0:
                raise TypeError(
                    f'Could not create {target_cls} from dict object.')
            elif len(concrete_cls) > 1:
                raise TypeError(f'More than one types in {target_cls} match '
                                'the given dict object.')
            else:
                return _create_dataclass_recursive(concrete_cls[0], obj)
        elif isinstance(obj, collections.abc.Sequence):
            # target cls should be a list (tuples are not supported)
            concrete_cls = [
                c for c in concrete_clss if typing.get_origin(c) == list
                and _check_list_compatibility(c, obj)
            ]
            if len(concrete_cls) == 0:
                raise TypeError(
                    f'Could not create {target_cls} from list object.')
            elif len(concrete_cls) > 1:
                if len(obj) == 0:
                    # empty list is an exception
                    return obj
                raise TypeError(f'More than one types in {target_cls} match '
                                'the given list object.')
            else:
                return concrete_cls[0](
                    _construct_dataclass(
                        typing.get_args(concrete_cls[0])[0],
                        elem,
                        allow_extra_keys=allow_extra_keys) for elem in obj)
        else:
            # elementary (non-compound) type
            for concrete_cls in concrete_clss:
                if typing.get_origin(concrete_cls) is not None:
                    continue
                if isinstance(obj, concrete_cls):
                    return obj
            raise TypeError(
                f'Could not create {target_cls} from object of type '
                f'{type(obj)}.')
    if not dataclasses.is_dataclass(target_cls):
        # Leaf node
        return target_cls(obj)
    assert typing.get_origin(target_cls) != tuple, \
        'Tuples are not supported for deserialisation'
    return _create_dataclass_recursive(
        target_cls, obj, allow_extra_keys=allow_extra_keys)

def _check_dataclass_compatibility(target_cls: type, obj: dict) -> bool:
    """
    Check whether the given dataclass type can be constructed from the given
    dict object

    Does a one level (shallow) check of the fields. It compares the field names
    to the dictionary keys. If the names of the necessary fields are present in
    the dictionary keys, and all the keys have their matching (potentially
    optional) fields, then it returns ``True``, otherwise ``False``.

    Considers only fields with ``init is True``.
    """
    assert dataclasses.is_dataclass(target_cls), 'Dataclass expected'
    fields = {f.name for f in dataclasses.fields(target_cls) if f.init}
    necessary_fields = {
        f.name
        for f in dataclasses.fields(target_cls)
        if f.init and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    keys = set(obj.keys())
    return necessary_fields.issubset(keys) and keys.issubset(fields)

def _check_list_compatibility(target_cls: type, obj: dict) -> bool:
    """
    Check whether the given list (generic) type can be constructed from the
    given list object.

    Currently, only nongeneric element types are supported.
    """
    assert typing.get_origin(target_cls) == list, 'List type expected'
    element_type = typing.get_args(target_cls)[0]
    assert typing.get_origin(element_type) is None, (
        'Only non-generic types are supported.')
    assert isinstance(obj, list), 'List expected for parameter `obj`'
    if len(obj) == 0:
        # empty list is compatible with everything
        return True
    return element_type == type(obj[0])

def _create_dataclass_recursive(target_cls: type[Dataclass], obj: dict, *,
                                allow_extra_keys: bool) -> Dataclass:
    """
    Creates a dataclass from the given dict.

    This differs from ``_construct_dataclass`` in that it expects ``target_cls``
    to be a dataclass.
    """
    assert dataclasses.is_dataclass(target_cls), 'Dataclass expected'
    fields = {f.name: f.type for f in dataclasses.fields(target_cls)}
    if allow_extra_keys:
        try:
            typed_obj = {
                fname:
                    _construct_dataclass(
                        ftype, obj[fname], allow_extra_keys=allow_extra_keys)
                for fname, ftype in fields.items()
            }
        except KeyError as e:
            raise TypeError(
                f'Missing entry "{e.args[0]}" from replay file; needed '
                f'to construct `{target_cls}`.') from e
    else:
        try:
            typed_obj = {
                k:
                    _construct_dataclass(
                        fields[k], v, allow_extra_keys=allow_extra_keys)
                for k, v in obj.items()
            }
        except KeyError as e:
            raise TypeError(f'Extra entry in replay file: "{e.args[0]}" for '
                            f'target class: {target_cls}') from e
    return target_cls(**typed_obj)

def deserialise(fname: str, *, allow_extra_keys: bool = False) -> Replay:
    with open(fname, 'r') as f:
        obj_dict = json.load(f)
    return _construct_dataclass(
        Replay, obj_dict, allow_extra_keys=allow_extra_keys)
