from train.arguments import Arguments
from train.preprocess import PreprocessPipeline
from train.train import TrainPipeline


def benchmark(args: Arguments) -> None:

    for task in args.tasks:
        print(f"######### Started evaluating {args.model_name} on task {task}")
        dataset = PreprocessPipeline().preprocess_dataset(args, task)

        training_pipeline = TrainPipeline(
            hulu_args=args, current_task=task, tokenizer_name=args.tokenizer_name
        )
        training_pipeline.set_tokenized_datasets(
            train_dataset=dataset["train"],
            dev_dataset=dataset["validation"],
            test_dataset=dataset["test"],
        )

        trained_model = training_pipeline.training()
        training_pipeline.create_submission(trained_model)
