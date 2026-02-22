# Durian Leaf Disease Classifier (Streamlit)

A Streamlit web app for durian leaf disease classification using the final research model (Transfer Learning).

## Project Structure

```text
Dur-HUB/
├─ app.py
├─ requirements.txt
├─ duren.ipynb
└─ artifacts/
   ├─ best_model_savedmodel/
   ├─ best_model.keras            # optional backup for local reuse
   ├─ class_names.npy
   └─ meta.json
```

## What Is Required for Deployment?

For **inference/deployment**, you only need:
- `app.py`
- `requirements.txt`
- `artifacts/` (model + labels + metadata)

You **do not need to upload the training dataset** to GitHub if the app is only used for prediction.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud (via GitHub)

1. Push this folder contents to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Set the main file path to:
   - `app.py`
5. Deploy.

## Notes

- The app loads the model from `artifacts/best_model_savedmodel`.
- `meta.json` is used to display model name and test accuracy, and to infer image input size.
- If your artifacts folder is stored elsewhere, you can set environment variable:
  - `DUREN_ARTIFACTS_DIR`

## Recommended GitHub Files (Optional)

You may also add:
- `.gitignore`
- `README.md` (this file)
- screenshots for project preview

## Troubleshooting

### App says artifacts are missing
Make sure these files exist in the repo:
- `artifacts/best_model_savedmodel/saved_model.pb`
- `artifacts/class_names.npy`
- `artifacts/meta.json`

### Deployment is too large
If GitHub rejects the model files because of size:
- use **Git LFS**, or
- store artifacts externally and download them at app startup (advanced setup)
