# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
FULL FLOW (what predict_falcipain2 does internally, step by step):

  1. read_input()                  -> normalizes loose SMILES or a CSV into a table
  2. validate_and_parse_smiles()   -> validates each SMILES with RDKit
  3. compute_mordred_descriptors() / compute_maccs_keys() -> compute descriptors
  4. align_features()              -> keeps the EXACT columns the models expect
  5. check_applicability_domain()  -> is this molecule "known" to the model?
  6. predict_with_top3()           -> runs the 3 best models and builds a consensus
  7. explain_prediction()          -> why active/inactive is predicted (SHAP)

Each function can be used on its own if its only needs one
step (e.g. just validating SMILES before sending them to the backend).
--------------------------------------------------------------------------
Requires: pandas, numpy, joblib, rdkit, mordred, scikit-learn, shap
"""

import io
import warnings
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors as mordred_descriptors

warnings.filterwarnings("ignore")

ARTIFACTS_DIR_DEFAULT = "artifacts"
LABEL_MEANING = {
    "1": "Active (IC50 < 5 uM against Falcipain-2)",
    "0": "Inactive (IC50 >= 5 uM against Falcipain-2)",
}


# =============================================================================
# STEP 1 - Read the user's input (a single SMILES, a list, or a CSV)
# =============================================================================

def read_input(data, input_type="smiles", smiles_column="smiles", id_column=None):
    """
    Parameters
    ----------
    data : str | list[str] | bytes | file-like
        - If input_type="smiles": one SMILES (str) or a list of SMILES.
        - If input_type="csv": the uploaded CSV content (bytes, a file
          path, or a file-like object). Must contain a column with the
          SMILES (named "smiles" by default).
    input_type : "smiles" | "csv"
    smiles_column : name of the SMILES column inside the CSV.
    id_column : name of a column to use as the row identifier (e.g.
        "compound_name"). If not given, an automatic id "mol_1",
        "mol_2", ... is generated.

    Returns
    -------
    pd.DataFrame with columns ["id", "smiles"]
    """
    if input_type == "smiles":
        smiles_list = [data] if isinstance(data, str) else list(data)
        ids = [f"mol_{i+1}" for i in range(len(smiles_list))]
        return pd.DataFrame({"id": ids, "smiles": smiles_list})

    elif input_type == "csv":
        if isinstance(data, (bytes, bytearray)):
            df = pd.read_csv(io.BytesIO(data))
        else:
            df = pd.read_csv(data) 
        if smiles_column not in df.columns:
            raise ValueError(
                f"The CSV has no column named '{smiles_column}'. "
                f"Columns found: {list(df.columns)}"
            )
        out = pd.DataFrame()
        out["id"] = df[id_column].astype(str) if id_column and id_column in df.columns \
            else [f"mol_{i+1}" for i in range(len(df))]
        out["smiles"] = df[smiles_column].astype(str)
        return out

    else:
        raise ValueError("input_type must be 'smiles' or 'csv'")


# =============================================================================
# STEP 2 - Validate SMILES with RDKit
# =============================================================================

def validate_and_parse_smiles(df):
    """
    Checks that each SMILES is chemically valid (RDKit can interpret it
    as an actual molecule).
    Input: DataFrame with columns ["id", "smiles"] (output of read_input).

    Returns
    -------
    df_valid : DataFrame ["id", "smiles"] with only the valid rows
    mols     : list of RDKit Mol objects, in the same order as df_valid
    errors   : list of dicts [{"id":..., "smiles":..., "reason":...}, ...]
               for rows that could NOT be parsed (to show as an error to
               the website's user).
    """
    valid_rows, mols, errors = [], [], []
    for _, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        mol = None
        if smi:
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None
        if mol is None:
            errors.append({"id": row["id"], "smiles": smi,
                            "reason": "Invalid SMILES or not parseable by RDKit"})
        else:
            valid_rows.append(row)
            mols.append(mol)
    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True) if valid_rows else \
        pd.DataFrame(columns=["id", "smiles"])
    return df_valid, mols, errors

# =============================================================================
# STEP 3a - Mordred descriptors (2D)
# =============================================================================

def compute_mordred_descriptors(mols):
    """
    Computes the Mordred 2D physicochemical descriptors for a list of
    already-validated RDKit molecules (output of validate_and_parse_smiles).

    Returns a DataFrame (one row per molecule, one column per descriptor).
    Can take a few seconds per molecule for large batches.
    """
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    df_desc = calc.pandas(mols, nproc=1, quiet=True)
    df_desc = df_desc.apply(pd.to_numeric, errors="coerce")
    return df_desc

# =============================================================================
# STEP 3b - MACCS structural keys
# =============================================================================

def compute_maccs_keys(mols):
    """
    Computes the MACCS fingerprint (166 structural bits, 0/1) for each
    molecule. 
    """
    rows = [list(MACCSkeys.GenMACCSKeys(m)) for m in mols]
    n_bits = len(rows[0]) if rows else 167
    cols = [f"MACCS_{i}" for i in range(n_bits)]
    return pd.DataFrame(rows, columns=cols)

# =============================================================================
# STEP 4 - Align computed descriptors with what the model expects
# =============================================================================

def align_features(df_raw, feature_columns):
    X = df_raw.reindex(columns=feature_columns)
    return X


# =============================================================================
# STEP 5 - Applicability domain (Standardization Approach)
# =============================================================================

def check_applicability_domain(X, bundle):
    """
    Checks whether each molecule is "within the applicability domain" (AD)
    of the model, i.e. whether it is similar enough to the molecules used
    for training for the prediction to be trusted.

    Method: Standardization Approach (Roy, Kar & Ambure, 2015), the same
    one used in the article. For each new compound:
        S_i = max_k | (descriptor_value_k - train_mean_k) / train_sd_k |
    compared against the SDC threshold computed during training.

    Parameters
    ----------
    X : already-aligned DataFrame (output of align_features) with the same
        columns as bundle["feature_columns"].
    bundle : dict loaded with load_artifact_bundle().

    Returns
    -------
    DataFrame with columns:
      - "S_i"      : the compound's maximum standardized distance (number)
      - "in_AD"    : True/False, whether it is within the applicability domain
      - "SDC"      : threshold used (same for every row, informational)
    """
    mean_train = bundle["ad_mean_train"]
    std_train = bundle["ad_std_train"]
    SDC = bundle["ad_SDC"]

    X_num = X.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(mean_train)  
    S_i = ((X_num - mean_train) / std_train).abs().max(axis=1)

    return pd.DataFrame({
        "S_i": S_i.values,
        "in_AD": (S_i <= SDC).values,
        "SDC": SDC,
    })

# =============================================================================
# Load the trained artifacts
# =============================================================================

def load_artifact_bundle(descriptor_type, artifacts_dir=ARTIFACTS_DIR_DEFAULT):
    """
    Loads the .joblib bundle
    descriptor_type : "mordred" | "maccs"
    """
    if descriptor_type not in ("mordred", "maccs"):
        raise ValueError("descriptor_type must be 'mordred' or 'maccs'")
    path = f"{artifacts_dir}/{descriptor_type}_bundle.joblib"
    return joblib.load(path)

# =============================================================================
# STEP 6 - Prediction with the 3 best models (consensus)
# =============================================================================

def predict_with_top3(X, bundle, active_threshold=0.5):
    """
    Runs the 3 best models stored in the bundle on X and combines their
    results.

    Returns a DataFrame (one row per molecule) with:
      - probability from each of the 3 individual models
        (columns "prob_<ModelName>")
      - "mean_probability" : average probability across the 3 models
      - "majority_vote"    : how many of the 3 models vote "Active"
      - "prediction"       : "Active" or "Inactive" 
      - "consensus"        : "unanimous" if all 3 models agree,
                              "majority" if only 2 of 3 agree
    """
    top3_models = bundle["top3_models"] 
    prob_cols = {}
    for model_name, pipeline in top3_models.items():
        probs = pipeline.predict_proba(X)[:, 1]
        prob_cols[f"prob_{model_name}"] = probs

    df_probs = pd.DataFrame(prob_cols)
    mean_probability = df_probs.mean(axis=1)
    votes_active = (df_probs >= 0.5).sum(axis=1)

    df_probs["mean_probability"] = mean_probability
    df_probs["majority_vote"] = votes_active
    df_probs["prediction"] = np.where(mean_probability >= active_threshold, "Active", "Inactive")
    df_probs["consensus"] = np.where(
        (votes_active == 3) | (votes_active == 0), "Unanimous", "Majority"
    )
    return df_probs

# =============================================================================
# STEP 7 - Per-molecule interpretability (SHAP)
# =============================================================================

def explain_prediction(X_query_row, bundle, model_name=None, top_n=10):
    """
    Explains WHY a specific molecule is predicted active or inactive:
    computes the contribution (SHAP) of each descriptor/MACCS bit to the
    prediction for that particular molecule.

    Parameters
    ----------
    X_query_row : single-row DataFrame, the molecule to explain.
    bundle : bundle loaded with load_artifact_bundle().
    model_name : which of the 3 models to use for the explanation (default:
        the first/best one in the ranking stored in the bundle).
    top_n : how many descriptors to show (the most influential ones).

    Returns
    -------
    DataFrame with columns:
      - "descriptor"        : name of the descriptor / MACCS bit
      - "molecule_value"    : value of that descriptor for the analyzed molecule
      - "contribution"      : SHAP value (how much this variable pushes
                               toward "Active" (+) or "Inactive" (-) for
                               THIS specific molecule)
      - "effect"             : "increases probability of activity" /
                                "decreases probability of activity"
    Sorted from most to least influential (absolute value).
    """
    import shap  

    if model_name is None:
        model_name = bundle["top3_ranking"][0]["model"]
    pipeline = bundle["top3_models"][model_name]
    feature_columns = bundle["feature_columns"]
    background = bundle["background_sample"][feature_columns]

    background_summary = shap.sample(background, min(30, len(background)), random_state=42)
    explainer = shap.Explainer(
        lambda data: pipeline.predict_proba(
            pd.DataFrame(data, columns=feature_columns))[:, 1],
        background_summary,
    )
    shap_values = explainer(X_query_row[feature_columns])

    contrib = pd.DataFrame({
        "descriptor": feature_columns,
        "molecule_value": X_query_row[feature_columns].values[0],
        "contribution": shap_values.values[0],
    })
    contrib["effect"] = np.where(
        contrib["contribution"] >= 0,
        "increases probability of activity",
        "decreases probability of activity",
    )
    contrib["abs_contribution"] = contrib["contribution"].abs()
    contrib = contrib.sort_values("abs_contribution", ascending=False) \
        .drop(columns="abs_contribution").head(top_n).reset_index(drop=True)
    contrib["model_used_for_explanation"] = model_name
    return contrib


# =============================================================================
# ORCHESTRATOR 
# =============================================================================

def predict_falcipain2(data, input_type="smiles", descriptor_types=("mordred", "maccs"),
                        artifacts_dir=ARTIFACTS_DIR_DEFAULT, explain=True,
                        smiles_column="smiles", id_column=None):
    """
    High-level function: takes whatever the website's user submitted
    (SMILES or CSV) and returns a list of results, one per molecule, ready
    to convert to JSON and show on the page.

    Parameters
    ----------
    data : whatever comes from the web form
    input_type : "smiles" | "csv"
    descriptor_types : which pipelines to run, a tuple with "mordred",
        "maccs" or both. If both are requested, each molecule carries a
        separate Mordred result and a separate MACCS result (they are two
        independent QSAR models, trained on different molecular
        information).
    artifacts_dir : folder containing mordred_bundle.joblib / maccs_bundle.joblib
    explain : if True, adds the SHAP interpretability for each molecule

    Returns
    -------
    dict with:
      "results": list of dicts (one per valid molecule), each with id,
          smiles, and for each requested descriptor_type:
              prediction, probability, within_applicability_domain,
              model_consensus, explanation (if explain=True)
      "errors": list of molecules that could not be processed (invalid
          SMILES), with the reason.
    """
    df_input = read_input(data, input_type=input_type,
                           smiles_column=smiles_column, id_column=id_column)
    df_valid, mols, errors = validate_and_parse_smiles(df_input)

    results = []
    if len(mols) > 0:
        raw_mordred = compute_mordred_descriptors(mols) if "mordred" in descriptor_types else None
        raw_maccs = compute_maccs_keys(mols) if "maccs" in descriptor_types else None

        bundles = {dt: load_artifact_bundle(dt, artifacts_dir) for dt in descriptor_types}

        per_type_output = {}
        for dt in descriptor_types:
            bundle = bundles[dt]
            raw = raw_mordred if dt == "mordred" else raw_maccs
            X = align_features(raw, bundle["feature_columns"])
            ad = check_applicability_domain(X, bundle)
            preds = predict_with_top3(X, bundle)
            per_type_output[dt] = {"X": X, "ad": ad, "preds": preds, "bundle": bundle}

        for i in range(len(df_valid)):
            row_result = {
                "id": df_valid.iloc[i]["id"],
                "smiles": df_valid.iloc[i]["smiles"],
            }
            for dt in descriptor_types:
                out = per_type_output[dt]
                pred_row = out["preds"].iloc[i]
                ad_row = out["ad"].iloc[i]
                bundle = out["bundle"]

                result_dt = {
                    "prediction": pred_row["prediction"],
                    "probability_active": round(float(pred_row["mean_probability"]), 4),
                    "model_consensus": pred_row["consensus"],
                    "votes_for_active": int(pred_row["majority_vote"]),
                    "within_applicability_domain": bool(ad_row["in_AD"]),
                    "applicability_domain_distance": round(float(ad_row["S_i"]), 4),
                    "label_meaning": LABEL_MEANING,
                    "models_used": [m["model"] for m in bundle["top3_ranking"]],
                }
                if explain:
                    X_row = out["X"].iloc[[i]]
                    expl = explain_prediction(X_row, bundle)
                    result_dt["top_descriptor_explanation"] = expl.to_dict("records")

                row_result[dt] = result_dt
            results.append(row_result)

    return {"results": results, "errors": errors}
