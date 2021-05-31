from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, TypeVar


@dataclass

class CustomTokens:

    START, END = '<', '>'
    _TEMPL = START + '{}' + END

    BOS_N   = _TEMPL.format('s')
    EOS_N   = _TEMPL.format('/s')
    START_N = _TEMPL.format('start')
    STOP_N  = _TEMPL.format('stop')
    PNTR_N  = _TEMPL.format('pointer')

    LIT_START = _TEMPL.format('lit')
    LIT_END   = _TEMPL.format('/lit')


    BOS_E   = _TEMPL.format('s')
    EOS_E   = _TEMPL.format('/s')
    START_E = _TEMPL.format('start')
    STOP_E  = _TEMPL.format('stop')

    _FIXED_SPECIAL_TOKENS_N = {
        BOS_N, EOS_N, START_N, STOP_N}
    _FIXED_SPECIAL_TOKENS_E = {
        BOS_E, EOS_E, START_E, STOP_E}
    _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E

    #SRL Tokens
    PRED = _TEMPL.format('P{}')

    @classmethod
    def is_node(cls, string: str) -> bool:
        if isinstance(string, str) and string.startswith(':'):
            return False
        elif string in cls._FIXED_SPECIAL_TOKENS_E:
            return False
        return True


T = TypeVar('T')

def index_default(
        item: T, list_: List[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        default: Optional[int] = None
):
    if start is None:
        start = 0
    if stop is None:
        stop = len(list_)
    return next((i for i, x in enumerate(list_[start:stop], start=start) if x == item), default)
