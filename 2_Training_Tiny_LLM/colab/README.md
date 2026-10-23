# Training on Google Colab from VS Code

These notebooks run on Colab's GPUs while you edit them in VS Code on the laptop. The 8 GB Mac stays free.

1. Install the **Google Colab** extension in VS Code.
2. Open a notebook, click **Select Kernel → Colab → New Colab Server**, sign in, and choose a GPU (T4 free; L4/A100 with Colab Pro).
3. Run the cells from the top. Each notebook clones the public
   [Kinthread-llm-model](https://github.com/ArAnirudh2901/Kinthread-llm-model) repo (code, data, checkpoints) and checks
   every file with `verify_model.py` first.

| Notebook | Does |
| --- | --- |
| `train_qa_colab.ipynb` | continues the relation model from `runs/clutrr-qa-v8/best.pt` (or trains from scratch), then evaluates end to end |
| `train_tagger_colab.ipynb` | trains and evaluates the name tagger |
| `export_and_verify_colab.ipynb` | exports chosen checkpoints to `web/model/` (exact fp32), rewrites and verifies `model_manifest.json` |

- **Results** go to Google Drive (`MyDrive/kinthread`) when Drive can be mounted. Otherwise they stay on the Colab
  machine until pushed. Colab wipes its disk when a session ends: free sessions last at most about 12 hours and end
  sooner when idle. Training writes `best.pt` as it improves, and `RESUME = True` continues a cut-off run.
- **Pushing** (last cell, optional) puts the results on a new `colab/...` branch of Kinthread-llm-model, never on
  `main`. It asks for a GitHub token and your commit email at run time and saves neither.
- **Precision** stays exact fp32: no mixed precision, no quantization.
- **The website** picks up a new model with `KINSHIP_SRC=<checkout of the branch> bun run sync`, then `bun run check`.
