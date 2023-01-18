from typing import Optional, List

import torch
from torch.utils.data import Dataset
from infohcvae.squad_utils import SquadExample, InputFeatures


class CustomDataset(Dataset):
    def __init__(self, all_c_ids: torch.Tensor, all_q_ids: torch.Tensor,
                 all_a_mask: torch.Tensor, all_start_mask: torch.Tensor, all_end_mask: torch.Tensor,
                 all_no_q_start_positions: torch.Tensor, all_no_q_end_positions: torch.Tensor,
                 is_train_set: bool = True, all_text_examples: Optional[List[SquadExample]] = None,
                 all_preprocessed_examples: Optional[List[InputFeatures]] = None, to_device: str="cpu"):
        self.num_items = all_c_ids.size(0)

        self.all_c_ids = all_c_ids
        self.all_q_ids = all_q_ids
        self.all_a_mask = all_a_mask
        self.all_start_mask = all_start_mask
        self.all_end_mask = all_end_mask
        self.all_no_q_start_positions = all_no_q_start_positions
        self.all_no_q_end_positions = all_no_q_end_positions
        self.indices = [[i] for i in range(self.num_items)]

        self.to_device = to_device

        self.all_text_examples = None
        self.all_preprocessed_examples = None
        if not is_train_set:
            assert all_text_examples is not None, "`all_text_examples` is required for dev set"
            assert all_preprocessed_examples is not None, "`all_preprocessed_examples` is required for dev set"

            self.all_text_examples = all_text_examples
            self.all_preprocessed_examples = all_preprocessed_examples

    def __getitem__(self, index):
        q_ids = self.all_q_ids[index]
        c_ids = self.all_c_ids[index]
        a_mask = self.all_a_mask[index]
        start_mask = self.all_start_mask[index]
        end_mask = self.all_end_mask[index]
        no_q_start_positions = self.all_no_q_start_positions[index]
        no_q_end_positions = self.all_no_q_end_positions[index]

        return self.indices[index], q_ids.to(self.to_device), c_ids.to(self.to_device),\
            a_mask.to(self.to_device), start_mask.to(self.to_device), end_mask.to(self.to_device),\
            no_q_start_positions.to(self.to_device), no_q_end_positions.to(self.to_device)

    def __len__(self):
        return self.num_items
