# Evaluation Example

This folder contains a minimal, end-to-end example to reproduce evaluation metrics on TIRAuxCloud pretrained models.

## Usage
1. Download the dataset and model weights from the TIRAuxCloud Hugging Face dataset page.
2. Populate the correct paths in `example_saved_model_run.json`.
3. Run the evaluation:
   ```bash
   python examples/run_example_evaluation.py --subset landsat