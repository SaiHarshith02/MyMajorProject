import os

# Suppress TF noise — must be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'model.tflite')
IMAGE_SIZE  = 128
LABELS      = ['[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B', 'Benign']
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# ── Medical information per class ────────────────────────────────────────────
MEDICAL_INFO = {
    '[Malignant] early Pre-B': {
        'full_name': 'Early Pre-B Cell Acute Lymphoblastic Leukemia (ALL)',
        'stage': 'Early Intermediate',
        'severity': 'High',
        'severity_level': 4,
        'urgency': 'Urgent',
        'urgency_desc': 'Prompt specialist consultation recommended within 24\u201348 hours',
        'color': '#f59e0b',
        'about': (
            'Early Pre-B ALL is a subtype of B-cell acute lymphoblastic leukemia '
            'where malignant cells are at an early intermediate stage of B-cell '
            'development, expressing cytoplasmic immunoglobulin heavy chains.'
        ),
        'symptoms': [
            'Persistent fatigue and unusual weakness',
            'Frequent or recurrent infections',
            'Easy bruising or unexplained bleeding',
            'Petechiae (tiny red spots under the skin)',
            'Bone or joint pain, especially in limbs',
            'Swollen lymph nodes in neck, armpits, or groin',
            'Unexplained recurring fever',
            'Night sweats',
            'Unintended weight loss',
            'Pale skin and shortness of breath',
        ],
        'suggestions': [
            'Consult a hematologist\u2011oncologist as soon as possible',
            'Request a Complete Blood Count (CBC) with differential and peripheral blood smear',
            'Undergo flow cytometry for immunophenotyping',
            'Schedule a bone marrow aspiration and biopsy for definitive diagnosis',
            'Request cytogenetic testing (karyotyping, FISH) for genetic markers',
            'Discuss molecular testing (PCR) for specific gene rearrangements',
            'Begin discussion of treatment protocols with your medical team',
            'Seek a second opinion from a specialized cancer center if needed',
        ],
    },
    '[Malignant] Pre-B': {
        'full_name': 'Pre-B Cell Acute Lymphoblastic Leukemia (ALL)',
        'stage': 'Intermediate',
        'severity': 'High',
        'severity_level': 4,
        'urgency': 'Urgent',
        'urgency_desc': 'Immediate specialist referral recommended',
        'color': '#f59e0b',
        'about': (
            'Pre-B ALL is characterized by the accumulation of malignant precursor '
            'B-cells expressing cytoplasmic \u03bc heavy chains. It is one of the most '
            'common subtypes of childhood ALL and also occurs in adults.'
        ),
        'symptoms': [
            'Severe and persistent fatigue',
            'Recurrent or serious infections',
            'Abnormal bleeding or easy bruising',
            'Petechiae or purpura on the skin',
            'Deep bone pain, particularly in legs and back',
            'Noticeably enlarged lymph nodes',
            'Hepatomegaly or splenomegaly (enlarged liver / spleen)',
            'High or persistent fever without clear cause',
            'Progressive weight loss and appetite loss',
            'Pallor and persistent shortness of breath',
        ],
        'suggestions': [
            'Seek immediate consultation with a hematologist\u2011oncologist',
            'Obtain CBC, LDH, uric acid, and coagulation studies',
            'Undergo bone marrow biopsy and aspiration',
            'Request comprehensive immunophenotyping via flow cytometry',
            'Pursue cytogenetic and molecular studies (BCR-ABL, MLL rearrangements)',
            'Discuss CNS evaluation (lumbar puncture) with your specialist',
            'Explore clinical trial availability for targeted therapy options',
            'Arrange psychosocial support and counseling services',
        ],
    },
    '[Malignant] Pro-B': {
        'full_name': 'Pro-B Cell Acute Lymphoblastic Leukemia (ALL)',
        'stage': 'Earliest / Most Immature',
        'severity': 'Critical',
        'severity_level': 5,
        'urgency': 'Emergency',
        'urgency_desc': 'Immediate emergency medical attention required',
        'color': '#ef4444',
        'about': (
            'Pro-B ALL represents the most immature form of B-cell leukemia. Blast '
            'cells are at the earliest identifiable stage of B-cell development and '
            'are often associated with MLL (KMT2A) gene rearrangements, requiring '
            'aggressive treatment.'
        ),
        'symptoms': [
            'Extreme fatigue and severe weakness',
            'High fever and serious, hard-to-treat infections',
            'Significant spontaneous bleeding or bruising',
            'Widespread petechiae across the body',
            'Intense, unrelenting bone pain',
            'Markedly swollen lymph nodes',
            'Abdominal distension from organ enlargement',
            'Rapid unexplained weight loss',
            'Drenching night sweats',
            'Severe anemia symptoms (dizziness, rapid heartbeat, fainting)',
        ],
        'suggestions': [
            'Seek emergency medical evaluation immediately',
            'Request comprehensive blood work \u2014 CBC, metabolic panel, coagulation profile',
            'Undergo urgent bone marrow biopsy and aspiration',
            'Rapid immunophenotyping and full genetic profiling',
            'Test specifically for MLL (KMT2A) gene rearrangements',
            'Begin treatment planning with a multidisciplinary oncology team',
            'Evaluate eligibility for clinical trials and novel therapies',
            'Arrange psychosocial support, patient advocacy, and caregiver resources',
        ],
    },
    'Benign': {
        'full_name': 'Benign \u2014 No Malignancy Detected',
        'stage': 'Normal',
        'severity': 'Normal',
        'severity_level': 1,
        'urgency': 'Routine',
        'urgency_desc': 'No immediate action required \u2014 continue routine monitoring',
        'color': '#22c55e',
        'about': (
            'The blood cell sample shows no signs of malignancy. Cells exhibit normal '
            'morphology consistent with healthy blood cells. This AI-based prediction '
            'should always be confirmed with professional laboratory analysis.'
        ),
        'symptoms': [
            'No cancer-related symptoms expected',
            'General good health indicators',
            'Normal blood cell morphology observed',
        ],
        'suggestions': [
            'Continue routine health check-ups as recommended by your physician',
            'Maintain regular blood work as part of annual physical examinations',
            'Report any new or unusual symptoms to your healthcare provider promptly',
            'Maintain a healthy lifestyle with balanced nutrition and regular exercise',
            'If symptoms persist despite benign results, request re-evaluation',
        ],
    },
}

# ── Load TFLite model ────────────────────────────────────────────────────────
print("Loading model …")
_interpreter = Interpreter(model_path=MODEL_PATH)
_interpreter.allocate_tensors()
_input_idx  = _interpreter.get_input_details()[0]['index']
_output_idx = _interpreter.get_output_details()[0]['index']
print("Model loaded.")


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def preprocess(image: Image.Image) -> np.ndarray:
    """Match training preprocessing exactly: resize with NEAREST, normalise [0,1]."""
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def infer(tensor: np.ndarray) -> np.ndarray:
    _interpreter.set_tensor(_input_idx, tensor)
    _interpreter.invoke()
    return _interpreter.get_tensor(_output_idx)[0]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/robots.txt')
def robots():
    return (
        "User-agent: *\nAllow: /\nDisallow: /predict\n"
        "Sitemap: https://bloodguard-ai.onrender.com/sitemap.xml\n"
    ), 200, {'Content-Type': 'text/plain'}


@app.route('/sitemap.xml')
def sitemap():
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        '<url><loc>https://bloodguard-ai.onrender.com/</loc>'
        '<changefreq>monthly</changefreq><priority>1.0</priority></url>'
        '</urlset>'
    ), 200, {'Content-Type': 'application/xml'}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file was uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html',
                               error='Unsupported file type. Upload a PNG, JPG, BMP, or TIFF image.')

    try:
        image  = Image.open(file.stream)
        tensor = preprocess(image)
        preds  = infer(tensor)

        class_idx  = int(np.argmax(preds))
        label      = LABELS[class_idx]
        confidence = round(float(preds[class_idx]) * 100, 2)
        info       = MEDICAL_INFO[label]

        all_probs = []
        for i, lbl in enumerate(LABELS):
            all_probs.append({
                'label': lbl,
                'prob': round(float(preds[i]) * 100, 2),
                'is_top': (i == class_idx),
            })
        all_probs.sort(key=lambda x: x['prob'], reverse=True)

        return render_template(
            'result.html',
            label=label,
            confidence=confidence,
            info=info,
            all_probs=all_probs,
            all_probs_json=json.dumps(all_probs),
            is_benign=(label == 'Benign'),
        )

    except Exception as e:
        return render_template('index.html', error=f'Prediction failed: {e}')


if __name__ == '__main__':
    app.run(debug=True)
