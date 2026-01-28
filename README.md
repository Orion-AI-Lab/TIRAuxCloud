# TIRAuxCloud Dataset

This supplementary material package accompanies the **CVPR submission** and is intended exclusively for review purposes.

---

## 📁 Contents

Inside the hugging face repository https://huggingface.co/datasets/tirauxcloud/TIRAuxCloud/tree/main the following are provided :

- Sample data from the **three test splits** of the three **TIRAuxCloud subsets**:
  - **Main Landsat**
  - **MA Landsat**
  - **VIIRS**

- Saved model weights used in the experiments presented in the paper (`model_files.tar.gz`)

Inside the supplementary material zip file are the code for reproducing the results of **Tables 2, 3, and 4** (running the models on the test datasets), along with the CSV files containing the corresponding model run parameters under `results` folder:
  - `models_table2.csv`
  - `models_table3.csv`
  - `models_table4.csv`

These files contain detailed information about the models that include:
  - model weight filenames  
  - metrics per class
  - hyperparameters  
  - input feature configurations  
  - and other metadata extracted from Weights & Biases

---

## 🔁 Reproducing Model Metrics

To reproduce the evaluation results of any model:

### 1️⃣ Complete the configuration file  
Edit **`saved_model_run.json`** and fill in the correct configuration block for the selected subset `"landsat"`, `"landsatMA"` or `"viirs"`

Make sure that all the parameters in the selected configuration block of the provided **`saved_model_run.json`** are filled correctly. No additional parameters are required beyond those already provided in **`saved_model_run.json`**

- `"dataset_folder"` parameter  is the folder of the taining file lists (csv files that define which are the training, validation and test files)
- `"dataset_dir"` is the folder of the dataset files (image samples)
- `"device"`, `"cpuworkers"` and `"batch_size"`** can be adjusted to your system's specifications

All the other parameters in the JSON configuration must match the corresponding values in the relevant CSV row of the selected model. All saved models are in the hugging face repository.

---

### 2️⃣ Run the evaluation script  
Execute **`model_test.py`** with the `-t` parameter to select the test subset:

```bash
python model_test.py -t landsat
python model_test.py -t landsatMA
python model_test.py -t viirs
