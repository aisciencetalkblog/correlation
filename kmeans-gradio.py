import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

state = {}

MAX_INPUTS = 20
input_components = []


def train_model(file, feature_cols, output_cols, n_clusters=8):
    if len(feature_cols) == 0 or len(output_cols) < 2:
        return "Select 1+ input features and at least 2 output columns.", gr.update(visible=False), gr.update(
            visible=False)

    try:
        data = pd.read_csv(file.name)
    except Exception as e:
        return f"Error reading file: {str(e)}", gr.update(visible=False), gr.update(visible=False)

    missing_features = set(feature_cols) - set(data.columns)
    missing_outputs = set(output_cols) - set(data.columns)
    if missing_features or missing_outputs:
        errors = []
        if missing_features:
            errors.append(f"Missing features: {missing_features}")
        if missing_outputs:
            errors.append(f"Missing outputs: {missing_outputs}")
        return "\n".join(errors), gr.update(visible=False), gr.update(visible=False)

    selected = feature_cols + output_cols
    data = data[selected].dropna()

    if len(data) < n_clusters:
        return f"Need at least {n_clusters} samples after cleaning", gr.update(visible=False), gr.update(visible=False)

    encoded_data = data.copy()
    label_encoders = {}
    for col in encoded_data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded_data[feature_cols])

    if np.unique(X_scaled, axis=0).shape[0] < n_clusters:
        return f"Not enough distinct points to form {n_clusters} clusters.", gr.update(visible=False), gr.update(
            visible=False)

    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(X_scaled)
    data['Cluster'] = kmeans.predict(X_scaled)

    output_maps, output_confs = {}, {}
    for col in output_cols:
        stats = (
            data.groupby(['Cluster', col])
            .size().reset_index(name='Count')
            .sort_values(['Cluster', 'Count'], ascending=[True, False])
        )
        target_map, target_conf = {}, {}
        for cluster in data['Cluster'].unique():
            top = stats[stats['Cluster'] == cluster].iloc[0]
            total = stats[stats['Cluster'] == cluster]['Count'].sum()
            target_map[cluster] = top[col]
            target_conf[cluster] = round(top['Count'] / total, 2)
        output_maps[col] = target_map
        output_confs[col] = target_conf

    state.update({
        "features": feature_cols,
        "output_cols": output_cols,
        "scaler": scaler,
        "kmeans": kmeans,
        "label_encoders": label_encoders,
        "output_maps": output_maps,
        "output_confs": output_confs,
        "data": data,
        "encoded_data": encoded_data
    })

    # Show only the needed number of input fields
    updates = []
    for i, comp in enumerate(input_components):
        if i < len(feature_cols):
            updates.append(gr.update(visible=True, label=feature_cols[i]))
        else:
            updates.append(gr.update(visible=False))

    return f"Model trained on {len(data)} samples.", gr.update(visible=True), gr.update(visible=True), *updates


def predict_new(*args):
    if not state:
        return {"error": "Model not trained yet."}

    features = state['features']
    args = args[:len(features)]  # Only take the active inputs
    if len(args) != len(features):
        return {"error": f"Expected {len(features)} features, got {len(args)}"}

    input_data = dict(zip(features, args))
    df = pd.DataFrame([input_data])

    for col, le in state['label_encoders'].items():
        if col in df.columns:
            mask = ~df[col].astype(str).isin(le.classes_)
            if mask.any():
                df.loc[mask, col] = le.classes_[0]
            df[col] = le.transform(df[col].astype(str))

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(state['encoded_data'][features].mean())

    scaled = state['scaler'].transform(df[features])
    cluster = state['kmeans'].predict(scaled)[0]

    result = {"Cluster": int(cluster)}
    for col in state['output_cols']:
        result[col] = state['output_maps'][col].get(cluster, "Unknown")
        result[f"Confidence_{col}"] = state['output_confs'][col].get(cluster, 0.0)
    return result


with gr.Blocks() as app:
    gr.Markdown("## 🔮 General KMeans Cluster Predictor")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            train_status = gr.Textbox(label="Training Status", interactive=False)
            feature_select = gr.Dropdown(label="Select Input Features", choices=[], multiselect=True)
            output_select = gr.Dropdown(label="Select Output Columns (2+ required)", choices=[], multiselect=True)
            n_clusters = gr.Slider(2, 20, value=8, step=1, label="Number of Clusters")
            train_button = gr.Button("Train Model")

        with gr.Column():
            with gr.Column(visible=False) as predict_section:
                gr.Markdown("### Prediction Inputs")
                for i in range(MAX_INPUTS):
                    box = gr.Textbox(label=f"Feature {i + 1}", visible=True)
                    input_components.append(box)
                predict_btn = gr.Button("Predict")
                prediction_output = gr.JSON()


    def update_column_choices(file):
        if file is None:
            return gr.update(choices=[]), gr.update(choices=[])
        try:
            df = pd.read_csv(file.name)
            cols = df.columns.tolist()
            return gr.update(choices=cols), gr.update(choices=cols)
        except:
            return gr.update(choices=[]), gr.update(choices=[])


    file_input.change(fn=update_column_choices, inputs=file_input, outputs=[feature_select, output_select])

    train_button.click(
        fn=train_model,
        inputs=[file_input, feature_select, output_select, n_clusters],
        outputs=[train_status, predict_section, predict_btn] + input_components
    )

    predict_btn.click(
        fn=predict_new,
        inputs=input_components,
        outputs=prediction_output
    )

app.launch(server_name="0.0.0.0", server_port=8080)








