import random
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, TensorDataset

from infohcvae.squad_utils import (convert_examples_to_features_answer_id,
                                   convert_examples_to_harv_features,
                                   read_squad_examples)
from infohcvae.model.custom.custom_torch_dataset import CustomDataset


def generate_testing_dataset_for_model_choosing(dataset: CustomDataset, batch_size=16, num_samples=100):
    assert dataset.all_preprocessed_examples is not None, "For debugging `all_preprocessed_examples` is required"
    assert dataset.all_text_examples is not None, "For debugging `all_text_examples` is required"

    all_c_ids = dataset.all_c_ids[:num_samples]
    all_q_ids = dataset.all_q_ids[:num_samples]
    all_a_mask = dataset.all_a_mask[:num_samples]
    all_no_q_start_positions = dataset.all_no_q_start_positions[:num_samples]
    all_no_q_end_positions = dataset.all_no_q_end_positions[:num_samples]
    all_text_examples = deepcopy(dataset.all_text_examples[:num_samples])
    all_preprocessed_examples = deepcopy(dataset.all_preprocessed_examples[:num_samples])

    rand_perm = torch.randperm(num_samples)

    shuffled_examples = [all_text_examples[idx] for idx in rand_perm.tolist()]
    shuffled_features = [all_preprocessed_examples[idx] for idx in rand_perm.tolist()]

    perm_dataset = CustomDataset(all_c_ids[rand_perm], all_q_ids[rand_perm],
                                 all_a_mask[rand_perm], all_no_q_start_positions[rand_perm],
                                 all_no_q_end_positions[rand_perm], is_train_set=False,
                                 all_text_examples=shuffled_examples,
                                 all_preprocessed_examples=shuffled_features)
    test_train_loader = DataLoader(perm_dataset, batch_size, shuffle=True)
    test_eval_loader = DataLoader(perm_dataset, batch_size, shuffle=False)

    return test_train_loader, test_eval_loader


def get_squad_data_loader(tokenizer, file, shuffle, is_train_set, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)

    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_context_length=args.max_c_len,
                                                      max_query_length=args.max_q_len,
                                                      doc_stride=128,
                                                      is_training=True)

    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_a_mask = (all_tag_ids != 0).long()
    all_start_mask = (all_tag_ids == 1).long()
    all_end_mask = (all_tag_ids == 3).long()
    all_no_q_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_no_q_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)

    all_data = CustomDataset(all_c_ids, all_q_ids, all_a_mask, all_start_mask, all_end_mask,
                             all_no_q_start_positions, all_no_q_end_positions, is_train_set=is_train_set,
                             all_text_examples=None if is_train_set else examples,
                             all_preprocessed_examples=None if is_train_set else features,
                             to_device=args.device)
    batch_size = args.batch_size if is_train_set else 64 # validation set batch_size=64 by default
    data_loader = DataLoader(all_data, batch_size, num_workers=args.num_workers, shuffle=shuffle)

    return data_loader, all_data


def get_harv_data_loader(tokenizer, file, shuffle, ratio, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    random.shuffle(examples)
    num_ex = int(len(examples) * ratio)
    examples = examples[:num_ex]
    features = convert_examples_to_harv_features(examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=args.max_c_len,
                                                 max_query_length=args.max_q_len,
                                                 doc_stride=128,
                                                 is_training=True)
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_c_ids)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=args.batch_size)

    return features, dataloader


def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_mask, start_mask, end_mask, start_positions, end_positions = batch

    # c_len = torch.sum(torch.sign(c_ids), 1)
    # max_c_len = torch.max(c_len)
    # c_ids = c_ids[:, :max_c_len]
    # a_ids = a_ids[:, :max_c_len]

    # q_len = torch.sum(torch.sign(q_ids), 1)
    # max_q_len = torch.max(q_len)
    # q_ids = q_ids[:, :max_q_len]

    return c_ids, q_ids, a_mask, start_mask, end_mask, start_positions, end_positions
