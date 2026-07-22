from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# ──────────────────────────────────────────────
#  TRAIN MODELS ON STARTUP
# ──────────────────────────────────────────────
models  = {}
scalers = {}

# ── 1. Breast Cancer (real sklearn dataset) ──
def train_breast():
    data = load_breast_cancer()
    X, y = data.data, data.target
    sc = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['breast']  = model
    scalers['breast'] = sc

# ── 2. Lung Cancer (synthetic) ──
def train_lung():
    np.random.seed(42); n = 1000
    X = np.column_stack([
        np.random.randint(30, 85, n),
        np.random.randint(0,  50, n),
        np.random.uniform(0,   3, n),
        np.random.randint(0,   2, n),
        np.random.randint(0,   2, n),
        np.random.randint(0,   2, n),
        np.random.randint(0,   2, n),
        np.random.randint(0,   2, n),
        np.random.randint(0,   2, n),
        np.random.randint(1,   5, n),
    ])
    risk = (X[:,0]/85*.2 + X[:,1]/50*.3 + X[:,2]/3*.15 +
            X[:,3]*.1  + X[:,4]*.05 + X[:,5]*.05 +
            X[:,6]*.05 + X[:,7]*.03 + X[:,8]*.04 + X[:,9]/5*.03)
    y = (risk + np.random.normal(0,.05,n) > 0.35).astype(int)
    sc = StandardScaler()
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['lung']  = model
    scalers['lung'] = sc

# ── 3. Diabetes / Pancreatic (synthetic) ──
def train_diabetes():
    np.random.seed(42); n = 1000
    X = np.column_stack([
        np.random.randint(0,  15, n),
        np.random.randint(70,200, n),
        np.random.randint(40,130, n),
        np.random.randint(0,  99, n),
        np.random.randint(0, 850, n),
        np.random.uniform(18, 45, n),
        np.random.uniform(.08,2.5,n),
        np.random.randint(21, 80, n),
    ])
    risk = (X[:,1]/200*.3 + X[:,5]/45*.2 + X[:,6]/2.5*.15 +
            X[:,7]/80*.15 + X[:,4]/850*.1 + X[:,2]/130*.1)
    y = (risk + np.random.normal(0,.05,n) > 0.4).astype(int)
    sc = StandardScaler()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['diabetes']  = model
    scalers['diabetes'] = sc

# ── 4. Skin Cancer (synthetic) ──
def train_skin():
    np.random.seed(42); n = 1000
    X = np.column_stack([
        np.random.uniform(0, 10, n),
        np.random.uniform(0, 10, n),
        np.random.randint(1,  6, n),
        np.random.uniform(1, 30, n),
        np.random.randint(0,  2, n),
        np.random.randint(1,  5, n),
        np.random.randint(0, 50, n),
        np.random.randint(0,  2, n),
        np.random.randint(20,80, n),
        np.random.randint(0, 50, n),
    ])
    risk = (X[:,0]/10*.2 + X[:,1]/10*.15 + X[:,2]/5*.1 +
            X[:,3]/30*.15 + X[:,4]*.1 + (5-X[:,5])/4*.05 +
            X[:,6]/50*.1  + X[:,7]*.05 + X[:,8]/80*.05 + X[:,9]/50*.05)
    y = (risk + np.random.normal(0,.05,n) > 0.35).astype(int)
    sc = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['skin']  = model
    scalers['skin'] = sc

# ── 5. Cervical Cancer (synthetic) ──
def train_cervical():
    np.random.seed(42); n = 1000
    X = np.column_stack([
        np.random.randint(15, 70, n),   # age
        np.random.randint(1,  20, n),   # sexual_partners
        np.random.randint(10, 25, n),   # first_intercourse_age
        np.random.randint(0,   5, n),   # pregnancies
        np.random.randint(0,   2, n),   # smokes
        np.random.randint(0,  30, n),   # smokes_years
        np.random.randint(0,   2, n),   # hormonal_contraceptives
        np.random.randint(0,  15, n),   # hc_years
        np.random.randint(0,   2, n),   # iud
        np.random.randint(0,   2, n),   # stds
    ])
    risk = (X[:,0]/70*.1 + X[:,1]/20*.2 + (25-X[:,2])/15*.1 +
            X[:,3]/5*.05 + X[:,4]*.15 + X[:,5]/30*.1 +
            X[:,6]*.1  + X[:,7]/15*.05 + X[:,8]*.05 + X[:,9]*.1)
    y = (risk + np.random.normal(0,.05,n) > 0.35).astype(int)
    sc = StandardScaler()
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['cervical']  = model
    scalers['cervical'] = sc

# ── 6. Prostate Cancer (synthetic) ──
def train_prostate():
    np.random.seed(42); n = 1000
    X = np.column_stack([
        np.random.randint(40, 85, n),   # age
        np.random.uniform(0,  10, n),   # psa_level
        np.random.uniform(0,   5, n),   # psa_density
        np.random.randint(0,   2, n),   # family_history
        np.random.randint(0,   2, n),   # african_american
        np.random.uniform(20,  45, n),  # bmi
        np.random.randint(0,   2, n),   # frequent_urination
        np.random.randint(0,   2, n),   # weak_urine_flow
        np.random.randint(0,   2, n),   # blood_in_urine
        np.random.randint(0,   2, n),   # erectile_dysfunction
    ])
    risk = (X[:,0]/85*.2 + X[:,1]/10*.3 + X[:,2]/5*.15 +
            X[:,3]*.1  + X[:,4]*.05 + X[:,5]/45*.05 +
            X[:,6]*.05 + X[:,7]*.04 + X[:,8]*.04 + X[:,9]*.02)
    y = (risk + np.random.normal(0,.05,n) > 0.35).astype(int)
    sc = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(sc.fit_transform(X), y)
    models['prostate']  = model
    scalers['prostate'] = sc

train_breast()
train_lung()
train_diabetes()
train_skin()
train_cervical()
train_prostate()
print("✅ All 6 models trained successfully.")

# ──────────────────────────────────────────────
#  FEATURE DEFINITIONS
# ──────────────────────────────────────────────
CANCER_FEATURES = {
    'breast': { 'name': 'Breast Cancer', 'features': [
        {'key':'mean_radius',            'label':'Mean Radius',            'min':6,    'max':30,   'step':0.1,   'default':14},
        {'key':'mean_texture',           'label':'Mean Texture',           'min':9,    'max':40,   'step':0.1,   'default':19},
        {'key':'mean_perimeter',         'label':'Mean Perimeter',         'min':40,   'max':190,  'step':0.1,   'default':92},
        {'key':'mean_area',              'label':'Mean Area',              'min':140,  'max':2500, 'step':1,     'default':655},
        {'key':'mean_smoothness',        'label':'Mean Smoothness',        'min':0.05, 'max':0.16, 'step':0.001, 'default':0.096},
        {'key':'mean_compactness',       'label':'Mean Compactness',       'min':0.02, 'max':0.35, 'step':0.001, 'default':0.104},
        {'key':'mean_concavity',         'label':'Mean Concavity',         'min':0,    'max':0.43, 'step':0.001, 'default':0.089},
        {'key':'mean_concave_points',    'label':'Mean Concave Points',    'min':0,    'max':0.2,  'step':0.001, 'default':0.049},
        {'key':'mean_symmetry',          'label':'Mean Symmetry',          'min':0.1,  'max':0.3,  'step':0.001, 'default':0.181},
        {'key':'mean_fractal_dimension', 'label':'Mean Fractal Dimension', 'min':0.05, 'max':0.1,  'step':0.001, 'default':0.063},
    ]},
    'lung': { 'name': 'Lung Cancer', 'features': [
        {'key':'age',              'label':'Age',                              'min':30,'max':85,'step':1,  'default':55},
        {'key':'smoking_years',    'label':'Smoking Years',                    'min':0, 'max':50,'step':1,  'default':10},
        {'key':'packs_per_day',    'label':'Packs Per Day',                    'min':0, 'max':3, 'step':0.1,'default':0.5},
        {'key':'coughing_blood',   'label':'Coughing Blood (0=No, 1=Yes)',     'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'chest_pain',       'label':'Chest Pain (0=No, 1=Yes)',         'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'short_breath',     'label':'Shortness of Breath (0=No, 1=Yes)','min':0,'max':1, 'step':1,  'default':0},
        {'key':'weight_loss',      'label':'Unexplained Weight Loss (0/1)',    'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'fatigue',          'label':'Fatigue (0=No, 1=Yes)',            'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'wheezing',         'label':'Wheezing (0=No, 1=Yes)',           'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'air_pollution',    'label':'Air Pollution Level (1–5)',        'min':1, 'max':5, 'step':1,  'default':2},
    ]},
    'diabetes': { 'name': 'Diabetes / Pancreatic Risk', 'features': [
        {'key':'pregnancies',      'label':'Pregnancies',               'min':0,    'max':15,  'step':1,    'default':2},
        {'key':'glucose',          'label':'Glucose Level (mg/dL)',     'min':70,   'max':200, 'step':1,    'default':110},
        {'key':'blood_pressure',   'label':'Blood Pressure (mmHg)',     'min':40,   'max':130, 'step':1,    'default':72},
        {'key':'skin_thickness',   'label':'Skin Thickness (mm)',       'min':0,    'max':99,  'step':1,    'default':23},
        {'key':'insulin',          'label':'Insulin (µU/mL)',           'min':0,    'max':850, 'step':1,    'default':79},
        {'key':'bmi',              'label':'BMI',                       'min':18,   'max':45,  'step':0.1,  'default':26},
        {'key':'pedigree',         'label':'Diabetes Pedigree Function','min':0.08, 'max':2.5, 'step':0.01, 'default':0.47},
        {'key':'age',              'label':'Age',                       'min':21,   'max':80,  'step':1,    'default':35},
    ]},
    'skin': { 'name': 'Skin Cancer', 'features': [
        {'key':'asymmetry',    'label':'Asymmetry Score (0–10)',         'min':0, 'max':10,'step':0.1,'default':2},
        {'key':'border',       'label':'Border Irregularity (0–10)',     'min':0, 'max':10,'step':0.1,'default':2},
        {'key':'color',        'label':'Color Variation (1–5)',          'min':1, 'max':5, 'step':1,  'default':2},
        {'key':'diameter',     'label':'Diameter (mm)',                  'min':1, 'max':30,'step':0.1,'default':5},
        {'key':'evolving',     'label':'Changed Recently (0=No, 1=Yes)', 'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'skin_type',    'label':'Fitzpatrick Skin Type (1–5)',    'min':1, 'max':5, 'step':1,  'default':3},
        {'key':'sun_exposure', 'label':'Sun Exposure (years)',           'min':0, 'max':50,'step':1,  'default':10},
        {'key':'family_hist',  'label':'Family History (0=No, 1=Yes)',   'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'age',          'label':'Age',                            'min':20,'max':80,'step':1,  'default':40},
        {'key':'moles',        'label':'Number of Moles',                'min':0, 'max':50,'step':1,  'default':10},
    ]},
    'cervical': { 'name': 'Cervical Cancer', 'features': [
        {'key':'age',              'label':'Age',                                    'min':15,'max':70,'step':1,'default':30},
        {'key':'partners',         'label':'Number of Sexual Partners',              'min':1, 'max':20,'step':1,'default':3},
        {'key':'first_intercourse','label':'Age at First Intercourse',               'min':10,'max':25,'step':1,'default':18},
        {'key':'pregnancies',      'label':'Number of Pregnancies',                  'min':0, 'max':5, 'step':1,'default':1},
        {'key':'smokes',           'label':'Smokes (0=No, 1=Yes)',                   'min':0, 'max':1, 'step':1,'default':0},
        {'key':'smokes_years',     'label':'Smoking Years',                          'min':0, 'max':30,'step':1,'default':0},
        {'key':'hormonal_contra',  'label':'Hormonal Contraceptives (0=No, 1=Yes)',  'min':0, 'max':1, 'step':1,'default':0},
        {'key':'hc_years',         'label':'Contraceptive Use (years)',              'min':0, 'max':15,'step':1,'default':0},
        {'key':'iud',              'label':'IUD (0=No, 1=Yes)',                      'min':0, 'max':1, 'step':1,'default':0},
        {'key':'stds',             'label':'STDs History (0=No, 1=Yes)',             'min':0, 'max':1, 'step':1,'default':0},
    ]},
    'prostate': { 'name': 'Prostate Cancer', 'features': [
        {'key':'age',              'label':'Age',                              'min':40,'max':85,'step':1,  'default':60},
        {'key':'psa_level',        'label':'PSA Level (ng/mL)',                'min':0, 'max':10,'step':0.1,'default':1.5},
        {'key':'psa_density',      'label':'PSA Density',                     'min':0, 'max':5, 'step':0.1,'default':0.5},
        {'key':'family_history',   'label':'Family History (0=No, 1=Yes)',     'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'african_american', 'label':'African American (0=No, 1=Yes)',   'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'bmi',              'label':'BMI',                              'min':20,'max':45,'step':0.1,'default':27},
        {'key':'freq_urination',   'label':'Frequent Urination (0=No, 1=Yes)', 'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'weak_urine_flow',  'label':'Weak Urine Flow (0=No, 1=Yes)',    'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'blood_in_urine',   'label':'Blood in Urine (0=No, 1=Yes)',     'min':0, 'max':1, 'step':1,  'default':0},
        {'key':'erectile_dysfunc', 'label':'Erectile Dysfunction (0=No, 1=Yes)','min':0,'max':1, 'step':1,  'default':0},
    ]},
}

# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────
@app.route('/api/cancer-types', methods=['GET'])
def get_cancer_types():
    return jsonify({'types': [
        {'id':'breast',   'name':'Breast Cancer',             'icon':'🎗️', 'color':'#e91e8c'},
        {'id':'lung',     'name':'Lung Cancer',               'icon':'🫁', 'color':'#2196F3'},
        {'id':'diabetes', 'name':'Diabetes / Pancreatic Risk','icon':'🩸', 'color':'#FF5722'},
        {'id':'skin',     'name':'Skin Cancer',               'icon':'🔬', 'color':'#FF9800'},
        {'id':'cervical', 'name':'Cervical Cancer',           'icon':'🩺', 'color':'#9C27B0'},
        {'id':'prostate', 'name':'Prostate Cancer',           'icon':'⚕️', 'color':'#00BCD4'},
    ]})

@app.route('/api/features/<cancer_type>', methods=['GET'])
def get_features(cancer_type):
    if cancer_type not in CANCER_FEATURES:
        return jsonify({'error': 'Cancer type not found'}), 404
    return jsonify(CANCER_FEATURES[cancer_type])

@app.route('/api/predict', methods=['POST'])
def predict():
    data        = request.get_json()
    cancer_type = data.get('cancer_type')
    features    = data.get('features', [])

    if cancer_type not in models:
        return jsonify({'error': 'Model not found'}), 404

    try:
        X        = np.array(features, dtype=float).reshape(1, -1)
        X_scaled = scalers[cancer_type].transform(X)
        model    = models[cancer_type]
        prob     = model.predict_proba(X_scaled)[0]
        risk     = round(float(prob[1]) * 100, 1)

        if risk < 30:
            level, color = 'Low',      '#22c55e'
            advice = 'Your risk indicators appear low. Continue regular check-ups and maintain a healthy lifestyle.'
        elif risk < 60:
            level, color = 'Moderate', '#f59e0b'
            advice = 'Moderate risk detected. Consider consulting a healthcare professional for further screening.'
        else:
            level, color = 'High',     '#ef4444'
            advice = 'High risk indicators detected. Please consult a medical professional as soon as possible.'

        top_features = []
        if hasattr(model, 'feature_importances_'):
            names = [f['label'] for f in CANCER_FEATURES[cancer_type]['features']]
            top_features = sorted(
                [{'feature': names[i], 'importance': round(float(v), 3)}
                 for i, v in enumerate(model.feature_importances_)],
                key=lambda x: x['importance'], reverse=True
            )[:5]

        return jsonify({
            'risk_score':   risk,
            'risk_level':   level,
            'risk_color':   color,
            'advice':       advice,
            'top_features': top_features,
            'disclaimer':   'Educational ML model only — NOT a substitute for medical diagnosis.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify({'datasets': [
        {'cancer':'Breast Cancer',        'name':'Wisconsin Breast Cancer',    'source':'UCI/sklearn','url':'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic','sklearn':'from sklearn.datasets import load_breast_cancer','kaggle':'https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data','samples':569,     'features':30,         'target':'Malignant/Benign'},
        {'cancer':'Lung Cancer',          'name':'Survey Lung Cancer',         'source':'Kaggle',     'url':'https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer',                                                                                               'samples':309,     'features':15,         'target':'Yes/No'},
        {'cancer':'Diabetes/Pancreatic',  'name':'Pima Indians Diabetes',      'source':'UCI/Kaggle', 'url':'https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database',                                                                                     'samples':768,     'features':8,          'target':'Diabetic/Non-Diabetic'},
        {'cancer':'Skin Cancer',          'name':'HAM10000 Skin Lesion',       'source':'ISIC/Kaggle','url':'https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection',                                                                    'samples':'10,015','features':'Image+meta','target':'7 lesion types'},
        {'cancer':'Cervical Cancer',      'name':'Cervical Cancer Risk',       'source':'UCI/Kaggle', 'url':'https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors',                                                                                     'samples':858,     'features':36,         'target':'Biopsy result'},
        {'cancer':'Prostate Cancer',      'name':'Prostate Cancer Dataset',    'source':'Kaggle',     'url':'https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer',                                                                                               'samples':100,     'features':9,          'target':'Diagnosis result'},
    ]})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(models.keys())})

if __name__ == '__main__':
    print("🚀 Starting Cancer Prediction API at http://localhost:5000")
    app.run(debug=True, port=5000)
