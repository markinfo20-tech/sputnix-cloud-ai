# SPUTNIX CLOUD AI

Streamlit Cloud deployment package.

## Deploy
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py`.
3. The first run downloads the CLIPSeg model from Hugging Face.

## Important limitations
This demo uses a ~603 MB model trained on Sentinel-2 / CloudSEN12+. Free hosting has memory, CPU, storage, and sleep limits; successful deployment and sustained inference are not guaranteed on a free tier. Validate model performance on Sputnix imagery before operational use. TIFF band order must also be confirmed.
