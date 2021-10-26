import logging
import os
import sys

import datasets
import jax
from flax.training.common_utils import shard
from jax import pmap
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, FlaxBertModel, TensorType
from transformers import (
    HfArgumentParser,
)

from tevatron.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.preprocessor import HFTestPreProcessor, HFCorpusPreProcessor

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = FlaxBertModel.from_pretrained(model_args.model_name_or_path, from_pt=True, config=config)

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_in_path:
        encode_dataset = EncodeDataset(data_args.encode_in_path, tokenizer, max_len=text_max_length)
        encode_dataset.encode_data = encode_dataset.encode_data \
            .shard(data_args.encode_num_shard, data_args.encode_shard_index)
    else:
        encode_dataset = datasets.load_dataset(data_args.dataset_name)[data_args.dataset_split] \
            .shard(data_args.encode_num_shard, data_args.encode_shard_index)
        processor = HFTestPreProcessor if data_args.encode_is_qry else HFCorpusPreProcessor
        encode_dataset = encode_dataset.map(
            processor(tokenizer, text_max_length),
            batched=False,
            num_proc=data_args.dataset_proc_num,
            remove_columns=encode_dataset.column_names,
            desc="Running tokenization",
        )
        encode_dataset = EncodeDataset(encode_dataset, tokenizer, max_len=text_max_length)

    total_batch_size = len(jax.devices()) * training_args.per_device_eval_batch_size
    padding_batch = {"text_id": [], "text": []}
    for i in range(total_batch_size - (len(encode_dataset) % total_batch_size)):
        padding_batch["text_id"].append(f"00000000000000{i}")
        padding_batch["text"].append([0])
    padding_batch = datasets.Dataset.from_dict(padding_batch)
    encode_dataset = datasets.concatenate_datasets([encode_dataset, padding_batch])

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size * len(jax.devices()),
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length',
            pad_to_multiple_of=16,
            return_tensors=TensorType.NUMPY,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    def encode_step(batch):
        embedding = model(**batch, train=False)[0]
        return embedding[:, 0]

    p_encode_step = pmap(encode_step)

    encoded = []
    lookup_indices = []

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        batch_embeddings = p_encode_step(shard(batch.data))
        encoded.extend(np.concatenate(batch_embeddings, axis=0))
    with open(data_args.encoded_save_path, 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

if __name__ == "__main__":
    main()
