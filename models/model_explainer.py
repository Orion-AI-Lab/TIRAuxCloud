import argparse
import os
import sys
import json
parent_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_script_dir)
from models_tcloud import init_model_and_loaders
import shap
import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)

def print_band_contributions(feature_names, band_importance_signed, band_importance_mag):
    max_len = max(len(f) for f in feature_names)

    print("\nBand Contribution (aligned):")
    print(f"{'Feature'.ljust(max_len)} | {'Signed':>10} | {'Magnitude':>10}")
    print("-" * (max_len + 27))

    for i, val in enumerate(band_importance_signed):
        fname = feature_names[i].ljust(max_len)
        sign_label = "+ve" if val >= 0 else "-ve"
        signed = f"{sign_label:>10}"
        mag = f"{band_importance_mag[i]:>10.4f}"
        print(f"{fname} | {signed} | {mag}")

def plot_pie_and_bar(feature_names, band_importance_mag, band_importance_signed):
    _, axs = plt.subplots(1, 2, figsize=(14, 6))

    wedges, _, _ = axs[0].pie(
        band_importance_mag,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.75,
        labeldistance=1.1,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 9}
    )

    axs[0].legend(
        wedges,
        feature_names,
        title="Bands",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9
    )
    axs[0].set_title("Band Importance (Magnitude)")
    axs[0].axis('equal')

    x_pos = np.arange(len(feature_names))
    axs[1].bar(x_pos, band_importance_signed)

    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(feature_names, rotation=45, ha='right')

    axs[1].set_title("Band Contribution (Signed SHAP)")
    axs[1].set_xlabel("Bands")
    axs[1].set_ylabel("Contribution")

    plt.tight_layout()
    plt.show()


def plot_waterfall(explanation, feature_names):
    shap.plots.waterfall(explanation, max_display=len(feature_names), show=True)

def evaluate_on_shap_explainer(params_dict, output_format=None, explainer_type="deep"):
    model, _, test_loader = init_model_and_loaders(params_dict)

    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    test_loader_iter = iter(test_loader)
    batch = next(test_loader_iter)
    images, _ = batch

    images = images.detach().cpu()
    bg_img_size = params_dict.get("bg_img_size", 100)
    test_img_size = params_dict.get("test_img_size", 3)

    if images.shape[0] < bg_img_size+test_img_size:
        bg_img_size = max(1, images.shape[0] - test_img_size) 

    indices = np.random.choice(images.shape[0], bg_img_size, replace=False)
    remaining_indices = np.setdiff1d(np.arange(images.shape[0]), indices)
    test_indices = remaining_indices[:test_img_size]
    background = images[indices]
    test_images = images[test_indices]

    # SHAP requires a scalar output, but segmentation models output HxW maps
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return out.mean(dim=(1,2,3)).unsqueeze(1)

    wrapped_model = ModelWrapper(model)

    device = params_dict.get("device", "cpu")
    wrapped_model.to(device)
    background = background.to(device)
    test_images = test_images.to(device)

    if explainer_type == "gradient":
        e = shap.GradientExplainer(wrapped_model, background)
        shap_values = e.shap_values(test_images, check_additivity=False)
    else:
        e = shap.DeepExplainer(wrapped_model, background)
        shap_values = e.shap_values(test_images, check_additivity=True)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_arr = shap_values.detach().cpu().numpy() if torch.is_tensor(shap_values) else shap_values

    if shap_arr.ndim == 5:
        shap_arr = shap_arr[..., 0]

    band_importance_signed = np.mean(shap_arr, axis=(0, 2, 3))
    band_importance_mag = np.abs(band_importance_signed)
    total = band_importance_mag.sum()
    if total == 0:
        band_importance_mag = np.zeros_like(band_importance_mag)
    else:
        band_importance_mag = band_importance_mag / total

    if output_format == "pie_and_bar":
        plot_pie_and_bar(params_dict['features'], band_importance_mag, band_importance_signed)

    print_band_contributions(params_dict['features'], band_importance_signed, band_importance_mag)

    feature_names = params_dict['features']

    if np.allclose(band_importance_signed, 0):
        scaled_values = band_importance_signed
    else:
        scale_factor = 1e8
        scaled_values = band_importance_signed * scale_factor

    explanation = shap.Explanation(
        values=scaled_values,
        base_values=0.0,
        data=scaled_values,
        feature_names=feature_names
    )

    if output_format is None or output_format == "waterfall":
        plot_waterfall(explanation, feature_names)

    return

def main():

    parser = argparse.ArgumentParser(
        description="Evaluation of models on Data set "
    )

    parser.add_argument(
        "-e", "--explain",
        type=str,
        choices=["viirs", "landsat", "landsatMA"],
        help="Data set identification (viirs, landsat, landsatMA)"
    )

    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["waterfall", "pie_and_bar"],
        default="waterfall",
        help="Format for DeepSHAP explainer function output visualization"
    )

    parser.add_argument(
        "--explainer","-m",
        type=str,
        choices=["deep", "gradient"],
        default="deep",
        help="SHAP explainer type"
    )

    args = parser.parse_args()
    test_set = args.explain
   
    configfile=os.path.join(script_dir,"configs/saved_models_run.json")
    with open(configfile, 'r') as file:
        configdict = json.load(file)
    if not test_set in configdict:
        print(f"{test_set} : Test set parameters not found")
    configdict=configdict[test_set]

    paramsdict = configdict["config"].copy()

    if "device" not in paramsdict:
        paramsdict["device"] = "cuda:0"

    if "traintest" not in paramsdict:
        paramsdict["traintest"] = "test"

    print("Starting Test. Model Parameters:")
    print(paramsdict)

    evaluate_on_shap_explainer(paramsdict, args.format, args.explainer)

if __name__ == "__main__":
    main()
