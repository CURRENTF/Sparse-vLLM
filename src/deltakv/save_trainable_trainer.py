from transformers.trainer import *

class SaveTrainableParamsTrainer(Trainer):
    def _collect_trainable_state_dict(self, state_dict=None):
        trainable_names = {name for name, param in self.model.named_parameters() if param.requires_grad}
        if state_dict is None:
            state_dict = self.model.state_dict()

        filtered_state_dict = {}
        for name, tensor in state_dict.items():
            if (
                name in trainable_names
                or (name.startswith("module.") and name[len("module."):] in trainable_names)
                or (f"module.{name}" in trainable_names)
            ):
                filtered_state_dict[name] = tensor.detach().cpu()
        return filtered_state_dict

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        unwrapped_model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(unwrapped_model, supported_classes):
            raise NotImplementedError
        else:
            trainable_state_dict = self._collect_trainable_state_dict(state_dict)
            unwrapped_model.save_pretrained(
                output_dir, state_dict=trainable_state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
                self.data_collator is not None
                and hasattr(self.data_collator, "tokenizer")
                and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
