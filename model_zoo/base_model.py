from abc import abstractmethod
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from preprocess.utils import *
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from rouge_score import *
import nltk
import numpy as np
import datasets
from transformers import Seq2SeqTrainer
class BaseModel:
    @abstractmethod
    def __init__(self, *args, **kwargs):

        self.max_input_length = 1024
        self.max_target_length = 128
        self.model_checkpoint = "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        self.metric = datasets.load_metric('rouge')

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]  # 按句子分割后换行符拼接
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    def preprocess_function(self, examples):

        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["summary"], max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    @abstractmethod
    def train(self):

        raw_data = load_xsum_data()
        tokenized_datasets = raw_data.map(self.preprocess_function, batched=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        batch_size = 64
        args = Seq2SeqTrainingArguments(
            "test-summarization",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,  # 至多保存3个模型
            num_train_epochs=1,
            predict_with_generate=True,
        )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
    @abstractmethod
    def summarize(self, *args, **kwargs):
        """
        TODO
        """
        pass
    
