# Deploy on Streamlit Community Cloud

This app is already in a GitHub repo. Use these steps to deploy.

## 1. Push your latest code (if needed)

```bash
git add .
git commit -m "Your message"
git push origin main
```

## 2. Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with **GitHub**.
2. Click **"New app"**.
3. Fill in:
   - **Repository:** `Yadav036/ADE-Classifier-Relational-Extraction` (or pick it from the list).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**.

Streamlit Cloud will build from this repo and run `streamlit run app.py`. It uses `requirements.txt` in the repo for dependencies.

## 3. If you see "not connected to a remote GitHub repository"

- That message usually means you’re trying to deploy from a **local path** or a **non-GitHub** source.
- Always choose **"GitHub"** as the source and select the repo **Yadav036/ADE-Classifier-Relational-Extraction** (and branch `main`) in the Cloud UI.
- Ensure your code is pushed: run `git push origin main` from this project folder.

## Notes

- **Secrets:** Use **Settings → Secrets** in the Cloud app to add env vars (e.g. if you use any API keys).
- **Models:** The app can download models from Google Drive at runtime (no need to store large files in the repo).
