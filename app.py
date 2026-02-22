import os
import json
import pathlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title='Durian Leaf Disease Classifier',
    page_icon='leaf',
    layout='centered'
)


def find_artifacts_dir():
    candidates = []
    env_dir = os.environ.get('DUREN_ARTIFACTS_DIR')
    if env_dir:
        candidates.append(env_dir)
    candidates.extend([
        './artifacts',
        'artifacts',
        '/mount/src/artifacts',
        '/app/artifacts',
    ])

    try:
        cwd = pathlib.Path('.').resolve()
        for p in [cwd] + list(cwd.iterdir()):
            if p.is_dir() and p.name.lower() == 'artifacts':
                candidates.append(str(p))
    except Exception:
        pass

    seen = set()
    for c in candidates:
        if not c:
            continue
        c = str(c)
        if c in seen:
            continue
        seen.add(c)
        export_dir = os.path.join(c, 'best_model_savedmodel')
        labels_path = os.path.join(c, 'class_names.npy')
        meta_path = os.path.join(c, 'meta.json')
        if os.path.exists(export_dir) and os.path.exists(labels_path) and os.path.exists(meta_path):
            return c
    return None


ARTIFACTS_DIR = find_artifacts_dir()
if ARTIFACTS_DIR is None:
    st.error('Folder artifacts tidak ditemukan atau tidak lengkap.')
    st.info('Pastikan tersedia: best_model_savedmodel/, class_names.npy, dan meta.json')
    st.stop()

EXPORT_DIR = os.path.join(ARTIFACTS_DIR, 'best_model_savedmodel')
LABELS_PATH = os.path.join(ARTIFACTS_DIR, 'class_names.npy')
META_PATH = os.path.join(ARTIFACTS_DIR, 'meta.json')


@st.cache_resource
def load_assets():
    model_obj = tf.saved_model.load(EXPORT_DIR)
    infer = model_obj.signatures['serving_default']
    class_names = np.load(LABELS_PATH, allow_pickle=True).tolist()
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return infer, class_names, meta


infer, class_names, meta = load_assets()
img_h, img_w = [int(v) for v in meta.get('img_size', [224, 224])]

st.title('Durian Leaf Disease Classifier')
acc_val = meta.get('best_model_accuracy', None)
model_name = meta.get('best_model_name', 'transfer')
if isinstance(acc_val, (int, float)) and float(acc_val) >= 0:
    st.caption(f"Model final riset: {model_name} | Test Accuracy: {float(acc_val):.4f}")
else:
    st.caption(f"Model final riset: {model_name} | Test Accuracy: N/A")

uploaded = st.file_uploader('Upload leaf image', type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Input Image', use_container_width=True)

    x = img.resize((img_w, img_h))
    x = np.array(x).astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    out_dict = infer(x_tf)
    probs = list(out_dict.values())[0].numpy()[0]
    pred_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: {class_names[pred_idx]}")
    st.write(f"Confidence: {float(probs[pred_idx]):.4f}")

    prob_rows = {
        'Class': class_names,
        'Probability': [float(p) for p in probs],
    }
    st.bar_chart(prob_rows, x='Class', y='Probability')
