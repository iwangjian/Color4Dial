# -*- coding: utf-8 -*-
import json
import dataclasses
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BrownianBridgeInput:
    """
    A single set of features of data for brownian bridge mapping.
    Property names are the same names as the corresponding inputs to a model.
    """
    user_utt_ids: List[int]
    follow_ids: List[int]
    transition_ids: List[int]
    interim_ids: List[int]
    start_ids: List[int]
    target_ids: List[int]
    interim_t: int
    target_T: int
    
    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class PlannerInput:
    """
    A single set of features of data for planning.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    decoder_input_ids_list: List[List[int]]
    decoder_input_all_ids: List[int]

    transition_ids: List[int]
    start_ids: List[int]
    target_ids: List[int]

    user_utt_ids: List[int]
    follow_ids: List[int]

    transition_number: int

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class DialogInput:
    """
    A single set of features of data for dialogue generation.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    lm_labels: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"