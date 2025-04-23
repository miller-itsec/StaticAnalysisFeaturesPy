import argparse
import os
import glob # Import glob for wildcard expansion
import pickle
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort # Import ONNX Runtime for validation
import pandas as pd
import shap
from onnxmltools import convert_lightgbm
from onnxsim import simplify
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(paths, feature_names, expected_dim=None):
    """
    Loads and combines data from multiple feature files.

    Args:
        paths (list): A list of paths to .json or .parquet files.
        feature_names (list): A list of feature names.
        expected_dim (int, optional): The expected feature dimension.
                                       If None, it's inferred from the first valid file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Combined features DataFrame with columns named according to feature_names.
            - np.ndarray: Combined labels array.
    """
    all_features_dfs = []
    all_labels = []
    inferred_dim = expected_dim # Use provided expected_dim initially
    first_file_processed = False

    print(f"📂 Loading data from {len(paths)} file(s)...")

    for path in paths:
        print(f"  -> Processing: {path}")
        if not os.path.exists(path):
            print(f"⚠️ File not found, skipping: {path}")
            continue

        try:
            if path.lower().endswith((".json", ".jsonl")):
                df = pd.read_json(path, lines=True)
            elif path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                print(f"⚠️ Unsupported file format, skipping: {path}")
                continue

            if "label" not in df.columns or "features" not in df.columns:
                print(f"⚠️ Missing 'label' or 'features' column, skipping: {path}")
                continue

            df["label"] = df["label"].astype(int)
            print("  🔍 Validating 'features' column...")

            # --- Feature Validation ---
            valid_mask = df["features"].apply(lambda x: isinstance(x, list) and all(isinstance(v, (int, float)) for v in x))
            dropped = (~valid_mask).sum()
            if dropped > 0:
                print(f"  ⚠️ Dropping {dropped} rows with invalid or non-numeric features in {path}")
                df = df[valid_mask]

            if df.empty:
                print(f"  ℹ️ No valid data left after filtering invalid features in {path}")
                continue

            df["features"] = df["features"].apply(lambda row: np.array(row, dtype=np.float32))

            # --- Dimension Check ---
            current_file_dim = len(df["features"].iloc[0])

            if not first_file_processed:
                # This is the first file with valid data
                if inferred_dim is None:
                    inferred_dim = current_file_dim
                    print(f"  ℹ️ Inferred feature dimension from first file ({path}): {inferred_dim}")
                elif inferred_dim != current_file_dim:
                     print(f"❌ Error: Feature dimension in first file ({path}) is {current_file_dim}, but expected {inferred_dim} (from --feature-names or prior file). Skipping file.")
                     continue # Skip this file

                # Validate feature_names length against inferred dimension
                if feature_names and len(feature_names) != inferred_dim:
                     print(f"❌ Error: Number of feature names ({len(feature_names)}) does not match inferred data dimension ({inferred_dim}).")
                     # Decide how to handle: raise error or try to adapt? Let's raise for now.
                     raise ValueError("Feature name count mismatch with data dimension.")

                first_file_processed = True # Mark that we have established the dimension

            # Check subsequent files against the established dimension
            if current_file_dim != inferred_dim:
                print(f"  ❌ Mismatched feature dimension in {path}. Expected: {inferred_dim}, Found: {current_file_dim}. Skipping file.")
                continue # Skip this file

            # Check for shape mismatches within the current file (redundant if first check passed, but safe)
            inconsistent_rows = df["features"].apply(lambda x: len(x) != inferred_dim)
            if inconsistent_rows.any():
                bad_count = inconsistent_rows.sum()
                print(f"  ❌ {bad_count} rows in {path} have incorrect feature dimensions. Expected: {inferred_dim}. Dropping these rows.")
                df = df[~inconsistent_rows]

            if df.empty:
                 print(f"  ℹ️ No valid data left after filtering inconsistent dimensions in {path}")
                 continue

            # --- Prepare for Concatenation ---
            features_array = np.stack(df["features"].values)
            labels_array = df["label"].values

            # Convert features to pandas DataFrame with names for this file
            if feature_names:
                features_df_single = pd.DataFrame(features_array, columns=feature_names)
            else:
                # Create generic names if none provided (should match inferred_dim)
                feature_names_generated = [f"feature_{i}" for i in range(inferred_dim)]
                features_df_single = pd.DataFrame(features_array, columns=feature_names_generated)
                if not first_file_processed: # Update global feature_names if generated on first file
                     feature_names = feature_names_generated


            all_features_dfs.append(features_df_single)
            all_labels.append(labels_array)
            print(f"  ✅ Loaded {features_array.shape[0]} samples from {path}")

        except FileNotFoundError:
             print(f"⚠️ File not found, skipping: {path}")
        except Exception as e:
            print(f"❌ Error processing file {path}: {e}. Skipping file.")
            # Consider adding more specific error handling if needed

    if not all_features_dfs:
        raise ValueError("❌ No valid data could be loaded from the provided input files.")

    # Combine data from all files
    print("\n🔄 Combining data from all loaded files...")
    combined_features_df = pd.concat(all_features_dfs, ignore_index=True)
    combined_labels = np.concatenate(all_labels)

    total_samples = combined_features_df.shape[0]
    final_dim = combined_features_df.shape[1]

    print(f"\n✅ Combined dataset loaded: {total_samples} samples, {final_dim} features")
    if final_dim != inferred_dim:
         print(f"⚠️ WARNING: Final dimension {final_dim} differs from inferred/expected dimension {inferred_dim}. Check data processing.")
    print(f"🔖 Unique labels in combined data: {np.unique(combined_labels)}")
    print("📊 Preview of first 5 combined feature vectors:")
    print(combined_features_df.head().to_string()) # Use to_string for better formatting

    # Return feature names used (could have been generated)
    final_feature_names = combined_features_df.columns.tolist()

    return combined_features_df, combined_labels, final_feature_names


def dump_weights(model, output_file="weights.bin"):
    print("💾 Saving raw model weights to: ", output_file)
    weights = model.feature_importances_
    weights.astype(np.float32).tofile(output_file)
    print("✅ Model weights saved to: ", output_file)


def add_missing_type_info(onnx_model):
    print("🔍 Checking and adding missing type information (explicit)...")
    made_changes = False

    # Use helper to create a float tensor type proto
    float_tensor_type_proto = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=None)

    # Check graph inputs
    for inp in onnx_model.graph.input:
        if not inp.type.tensor_type.elem_type:
            print(f"⚠️ Missing elem_type for input: {inp.name}. Setting to FLOAT.")
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            # Optionally clear shape if it causes issues, but usually keep it
            # for inp_dim in inp.type.tensor_type.shape.dim:
            #     pass # Keep existing shape info if present
            made_changes = True

    # Check graph outputs
    for outp in onnx_model.graph.output:
        if not outp.type.tensor_type.elem_type:
             # Handle sequence type if necessary (as in your original code)
            if outp.type.WhichOneof('value') == 'sequence_type':
                 sequence_type = outp.type.sequence_type
                 if not sequence_type.elem_type.tensor_type.elem_type:
                      print(f"⚠️ Missing elem_type for sequence output: {outp.name}. Setting to FLOAT.")
                      sequence_type.elem_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                      made_changes = True
            elif not outp.type.tensor_type.elem_type: # Standard tensor output
                 print(f"⚠️ Missing elem_type for output: {outp.name}. Setting to FLOAT.")
                 outp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                 made_changes = True

    # Check value_info (intermediate tensors)
    for vi in onnx_model.graph.value_info:
        if not vi.type.tensor_type.elem_type:
            print(f"⚠️ Missing elem_type for value_info: {vi.name}. Setting to FLOAT.")
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            # Keep potential shape info if present in vi.type.tensor_type.shape
            made_changes = True

    if made_changes:
        print("🔧 Type information was added/modified.")
    else:
        print("✅ No missing type information found to add.")

    # Validate the ONNX model after potential modifications
    try:
        # Use full_check for potentially more thorough validation
        onnx.checker.check_model(onnx_model, full_check=True)
        print("✅ ONNX model check passed after adding missing types (full_check).")
    except onnx.checker.ValidationError as e:
        print(f"❌ ONNX model check failed after modifications: {e}")
        # Consider saving the model here too for debugging the failed check
        # onnx.save(onnx_model, "model_failed_check.onnx")
        raise
    except Exception as e:
        print(f"❌ An unexpected error occurred during ONNX model check: {e}")
        raise

    return onnx_model


def train_and_export_onnx(X, y, output_path, feature_names, hyperparameters=None, no_wait=False):
    print("\n🧠 Training LightGBM model...")

    if not isinstance(X, pd.DataFrame):
        print("⚠️ Input X is not a Pandas DataFrame. Converting...")
        if feature_names and len(feature_names) == X.shape[1]:
             X = pd.DataFrame(X, columns=feature_names)
        else:
             # Generate generic feature names if conversion needed and names missing/mismatched
             print("⚠️ Generating generic feature names for DataFrame conversion.")
             generic_names = [f"feature_{i}" for i in range(X.shape[1])]
             X = pd.DataFrame(X, columns=generic_names)
             feature_names = generic_names # Use these names going forward


    # Feature scaling (standardizing the features)
    print("🔧 Scaling features using StandardScaler...")
    scaler = StandardScaler()
    # Fit scaler on the entire dataset (common practice before split, though fitting only on train is stricter)
    X_scaled = scaler.fit_transform(X)

    # Dump the scaler to a file
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved to 'scaler.pkl'")

    # Convert the scaler to JSON format
    scaler_data = {
        'mean': scaler.mean_.tolist(),  # Convert numpy array to list
        'scale': scaler.scale_.tolist(), # Convert numpy array to list
    }

    # Dump the scaler data to a JSON file
    with open('scaler.json', 'w') as json_file:
        json.dump(scaler_data, json_file, indent=4)
    print("✅ Scaler data saved to 'scaler.json'")

    # Train-test split
    print("🔪 Splitting data into training and testing sets (80/20)...")
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y # Stratify for potentially imbalanced labels
    )

    # Convert scaled data back to DataFrame with feature names for LightGBM compatibility
    # and for plotting feature importances correctly.
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    print(f"📊 Training set size: {X_train_df.shape}, Test set size: {X_test_df.shape}")

    # Initialize LightGBM model
    model = lgb.LGBMClassifier(objective='binary', random_state=42) # Add random_state for reproducibility
    print(f"🧬 Initial model parameters: {model.get_params()}")

    # Perform hyperparameter tuning if requested
    if hyperparameters:
        print("⚙️ Performing hyperparameter tuning using GridSearchCV...")
        grid_search = GridSearchCV(model, param_grid=hyperparameters, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_df, y_train) # Use DataFrames here
        print(f"🏆 Best hyperparameters found: {grid_search.best_params_}")
        print(f"📈 Best ROC AUC score during tuning: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_ # Use the best model found
    else:
        print("▶️ Fitting model with default/initial parameters...")
        # Add early stopping for efficiency if not tuning
        model.fit(X_train_df, y_train,
                  eval_set=[(X_test_df, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(10, verbose=True)]) # Stop if AUC on test set doesn't improve for 10 rounds


    # Model evaluation on the test set
    print("\n📈 Evaluating model performance on the test set...")
    y_pred_proba = model.predict_proba(X_test_df)[:, 1] # Probabilities for the positive class
    y_pred_binary = (y_pred_proba > 0.5).astype(int) # Binary predictions based on 0.5 threshold

    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"🎯 Test Set Accuracy: {accuracy:.4f}")
    print(f"📈 Test Set ROC AUC: {roc_auc:.4f}")

    # Save raw weights (feature importances)
    dump_weights(model, "weights.bin")

    # Plotting
    plot_feature_importance(model, feature_names, no_wait=no_wait)
    # Pass the original unscaled training data (or a sample) to SHAP for better interpretation
    # But ensure it has the correct feature names
    plot_shap_summary(model, X_train_df, feature_names, no_wait=no_wait) # Use scaled data for consistency with training


    # --- ONNX Conversion ---
    print("\n🔄 Converting LightGBM model to ONNX format...")

    # Define initial type based on the *scaled* data dimension
    initial_type = [('float_input', FloatTensorType([None, X_scaled.shape[1]]))]

    zipmap_option = True # Default is True (ZipMap enabled)
    try:
        # Simpler check for classifier type
        if isinstance(model, lgb.LGBMClassifier):
            zipmap_option = False # Disable ZipMap for classifiers for simpler output
            print("🔧 Model is LGBMClassifier, attempting conversion with zipmap=False (disabled).")
        else:
            print("ℹ️ Model is not LGBMClassifier, using default zipmap=True (enabled).")
    except Exception as e:
        print(f"⚠️ Error determining zipmap option, proceeding with default (zipmap=True): {e}")

    target_opset = 13 # Keep opset 13 unless specific compatibility issues arise

    try:
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_type,
            target_opset=target_opset,
            zipmap=zipmap_option
        )
        print(f"✅ Conversion successful (target_opset={target_opset}, zipmap={zipmap_option}).")
    except Exception as e:
        print(f"❌ Error during ONNX conversion (zipmap={zipmap_option}, opset={target_opset}): {e}")
        # Consider trying with zipmap flipped if conversion fails
        # if zipmap_option: # If it failed with True, try False
        #     try:
        #         print("🔧 Retrying conversion with zipmap=False...")
        #         onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=target_opset, zipmap=False)
        #         print("✅ Conversion successful on retry with zipmap=False.")
        #     except Exception as e2:
        #          print(f"❌ Error during ONNX conversion retry (zipmap=False): {e2}")
        #          raise e # Raise the original error
        # else: # If it failed with False, maybe raise directly
        raise # Re-raise the original error if conversion fails


    # Add missing type information if needed
    onnx_model = add_missing_type_info(onnx_model)

    # Simplify the ONNX model
    print("🔪 Simplifying the ONNX model...")
    try:
        model_simplified, check = simplify(onnx_model)
        if check:
            print("✅ ONNX model simplification successful.")
            onnx.save(model_simplified, output_path)
            print(f"✅ Simplified ONNX model saved to: {output_path}")
        else:
            print("⚠️ ONNX model simplification check failed. Saving the original model instead.")
            onnx.save(onnx_model, output_path) # Save the unsimplified one
            print(f"✅ Original (unsimplified) ONNX model saved to: {output_path}")
            model_simplified = onnx_model # Use the original for validation

    except Exception as e:
        print(f"❌ Error during ONNX model simplification: {e}. Saving the original model.")
        onnx.save(onnx_model, output_path) # Save the unsimplified one
        print(f"✅ Original (unsimplified) ONNX model saved to: {output_path}")
        model_simplified = onnx_model # Use the original for validation


    # Validate the final ONNX model using the scaled test data
    validate_onnx_model(model_simplified, X_test_df, y_test) # Pass scaled test DataFrame and labels


def validate_onnx_model(onnx_model, X_test_df, y_test=None):
    """Validates the ONNX model using ONNX Runtime."""
    print("\n🔍 Validating the ONNX model with ONNX Runtime...")

    if not isinstance(X_test_df, pd.DataFrame):
        print("❌ Error: Input X_test_df for validation must be a Pandas DataFrame.")
        return
    if X_test_df.isnull().values.any():
        print("❌ Error: Input X_test_df contains NaN values.")
        return


    try:
        # Create the ONNX runtime session
        session = ort.InferenceSession(onnx_model.SerializeToString())

        # Get the input name from the model
        input_name = session.get_inputs()[0].name
        print(f"  Model Input Name: '{input_name}'")
        print(f"  Model Input Shape Expected: {session.get_inputs()[0].shape}")
        print(f"  Validation Data Shape: {X_test_df.shape}")

        # Ensure data type is float32
        X_test_np = X_test_df.astype(np.float32).to_numpy()

        # Prepare input dictionary
        input_data = {input_name: X_test_np}

        # Perform inference
        onnx_outputs = session.run(None, input_data)
        print(f"  ONNX model produced {len(onnx_outputs)} output(s).")

        # Output analysis depends on whether zipmap was used
        # If zipmap=False (typical for classifiers), output is usually [labels, probabilities]
        # If zipmap=True, output might be different (often a single tensor if not classifier?)
        # Let's assume zipmap=False for classifier based on our conversion logic
        if len(onnx_outputs) == 2: # Likely [labels, probabilities]
             onnx_pred_labels = onnx_outputs[0]
             # Probabilities often come as a list of dictionaries like [{'0': prob0, '1': prob1}, ...]
             # We need to extract the probability for the positive class (label 1)
             try:
                 onnx_pred_proba = np.array([p[1] for p in onnx_outputs[1]])
                 print(f"  Output 0 (Labels) shape: {onnx_pred_labels.shape}")
                 print(f"  Output 1 (Probabilities) extracted shape: {onnx_pred_proba.shape}")
             except (TypeError, KeyError, IndexError) as e:
                  print(f"❌ Error extracting probabilities from ONNX output[1]: {e}")
                  print(f"  Raw output[1] sample: {onnx_outputs[1][:5]}") # Print sample for debugging
                  # Fallback: Maybe output[0] already contains probabilities if conversion was different?
                  if len(onnx_outputs[0].shape) == 1 or onnx_outputs[0].shape[1] == 1:
                     print("  ⚠️ Attempting to use output[0] as probabilities...")
                     onnx_pred_proba = onnx_outputs[0].flatten()
                  else:
                     print("❌ Cannot determine probability output.")
                     return # Cannot proceed with metrics

        elif len(onnx_outputs) == 1: # Might be probabilities directly (less common for classifiers with zipmap=False)
            print("  ⚠️ Single output detected. Assuming it contains probabilities.")
            onnx_pred_proba = onnx_outputs[0]
            # If shape is (N, 2), take prob of class 1. If (N,) or (N,1) assume it's prob of class 1.
            if len(onnx_pred_proba.shape) == 2 and onnx_pred_proba.shape[1] == 2:
                print("  Detected shape (N, 2), taking probabilities for class 1.")
                onnx_pred_proba = onnx_pred_proba[:, 1]
            elif len(onnx_pred_proba.shape) == 2 and onnx_pred_proba.shape[1] == 1:
                 onnx_pred_proba = onnx_pred_proba.flatten() # Make it 1D
            elif len(onnx_pred_proba.shape) == 1:
                 pass # Already 1D
            else:
                 print(f"❌ Unexpected shape for single output: {onnx_pred_proba.shape}")
                 return

            # Generate binary labels from probabilities if only probabilities are output
            onnx_pred_labels = (onnx_pred_proba > 0.5).astype(int)
            print(f"  Output 0 (Probabilities) shape: {onnx_pred_proba.shape}")


        else:
            print(f"❌ Unexpected number of outputs ({len(onnx_outputs)}) from ONNX model.")
            return

        print(f"✅ ONNX model inference successful!")
        print(f"  Sample Probabilities (ONNX): {onnx_pred_proba[:5]}")
        print(f"  Sample Predicted Labels (ONNX): {onnx_pred_labels[:5]}")


        # If y_test is provided, calculate and compare metrics
        if y_test is not None:
            print("\n📊 Comparing ONNX predictions with test labels...")
            try:
                accuracy_onnx = accuracy_score(y_test, onnx_pred_labels)
                roc_auc_onnx = roc_auc_score(y_test, onnx_pred_proba)
                print(f"  🎯 Accuracy (ONNX vs True): {accuracy_onnx:.4f}")
                print(f"  📈 ROC AUC (ONNX vs True): {roc_auc_onnx:.4f}")
            except Exception as e:
                print(f"❌ Error calculating metrics: {e}")
                print(f"   y_test shape: {y_test.shape}, type: {type(y_test)}")
                print(f"   onnx_pred_labels shape: {onnx_pred_labels.shape}, type: {type(onnx_pred_labels)}")
                print(f"   onnx_pred_proba shape: {onnx_pred_proba.shape}, type: {type(onnx_pred_proba)}")


        # Basic shape validation
        if X_test_df.shape[0] != onnx_pred_proba.shape[0]:
            print(f"⚠️ Warning: ONNX output has a different number of samples ({onnx_pred_proba.shape[0]}) than input ({X_test_df.shape[0]}).")

    except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e:
         print(f"❌ ONNX Runtime InvalidArgument Error: {e}")
         print("   This often means a mismatch between the model's expected input shape/type and the data provided.")
         # Add more debugging info if possible
         print(f"   Data type provided: {X_test_np.dtype}")
         input_details = session.get_inputs()[0]
         print(f"   Model expects Type: {input_details.type}, Shape: {input_details.shape}")
    except Exception as e:
        print(f"❌ Error during ONNX model validation/inference: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # raise # Optional: re-raise the exception



def plot_feature_importance(model, feature_names, output_file="feature_importance.txt", top_k=50, no_wait=False):
    print("📊 Plotting feature importance...")

    # Get feature importances from the trained model
    try:
        feature_importance = model.feature_importances_
    except AttributeError:
         print("⚠️ Could not get 'feature_importances_' from the model. Skipping plot.")
         return


    if feature_names is None or len(feature_names) == 0:
         print("⚠️ No feature names provided. Generating generic names for importance plot.")
         feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]


    if len(feature_names) != len(feature_importance):
        print(f"⚠️ Warning: Feature name count ({len(feature_names)}) doesn't match importance length ({len(feature_importance)}). Truncating/padding names.")
        if len(feature_names) < len(feature_importance):
            feature_names.extend([f"Unknown_{i}" for i in range(len(feature_names), len(feature_importance))])
        else:
            feature_names = feature_names[:len(feature_importance)]

    # Create pairs of (name, importance)
    feat_imp_pairs = list(zip(feature_names, feature_importance))

    # Filter out zero-importance features *before* sorting and saving
    filtered_pairs = [(name, imp) for name, imp in feat_imp_pairs if imp > 0]
    print(f"  Found {len(filtered_pairs)} features with importance > 0 (out of {len(feat_imp_pairs)} total).")

    if not filtered_pairs:
        print("  No features with positive importance found. Skipping plot and file output.")
        return

    # Sort by importance ascending
    sorted_features = sorted(filtered_pairs, key=lambda x: x[1])

    # Save ALL non-zero feature importances to file
    try:
        with open(output_file, "w") as f:
            # Save sorted descending (most important first)
            for name, importance in reversed(sorted_features):
                f.write(f"{name}: {importance}\n")
        print(f"✅ Non-zero feature importances saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving feature importance file: {e}")


    # Select top K (or fewer if less available) for plotting
    actual_top_k = min(top_k, len(sorted_features))
    top_features = sorted_features[-actual_top_k:]
    names = [name for name, _ in top_features]
    values = [importance for _, importance in top_features]

    # Plot
    plt.figure(figsize=(14, max(6, int(actual_top_k * 0.3)))) # Adjust height dynamically
    plt.barh(range(len(values)), values, align="center")
    plt.yticks(range(len(values)), names)
    plt.xlabel("Feature Importance (LGBM Gain/Split)")
    plt.title(f"Top {actual_top_k} Feature Importances (Importance > 0)")
    plt.tight_layout()
    importance_plot_file = "feature_importance_plot.png"
    try:
         plt.savefig(importance_plot_file)
         print(f"✅ Feature importance plot saved to: {importance_plot_file}")
    except Exception as e:
         print(f"❌ Error saving feature importance plot: {e}")

    if plt.get_backend().lower() != 'agg': # Don't try to show if using non-interactive backend
         plt.show(block=not no_wait)
         if no_wait:
             plt.pause(5) # Show for 5 seconds if no_wait
    plt.close() # Close the figure


def plot_shap_summary(model, X_df, feature_names, output_file="shap_summary.png", max_samples=1000, no_wait=False, num_features_to_show=30):
    print("\n📊 Generating SHAP summary plot...")

    if not isinstance(X_df, pd.DataFrame):
        print("⚠️ Input X_df for SHAP is not a Pandas DataFrame. Skipping SHAP plot.")
        return

    # Subsample if too many rows for performance
    if X_df.shape[0] > max_samples:
        print(f"  Subsampling from {X_df.shape[0]} to {max_samples} rows for SHAP calculation...")
        X_sample = shap.sample(X_df, max_samples, random_state=42) # Use shap's sampling
    else:
        X_sample = X_df

    print(f"  Calculating SHAP values for {X_sample.shape[0]} samples...")
    try:
        # Use TreeExplainer for tree models like LightGBM, it's faster
        explainer = shap.TreeExplainer(model)
        # SHAP values for binary classification often have two outputs (one per class)
        # We usually plot the SHAP values for the positive class (index 1)
        shap_values_obj = explainer(X_sample) # Use the explainer object callable interface

        # Access shap_values for the positive class if it's a multi-output explanation
        if isinstance(shap_values_obj.values, list) and len(shap_values_obj.values) > 1:
             shap_values_for_plot = shap_values_obj.values[1] # Assuming index 1 is positive class
             base_values_for_plot = shap_values_obj.base_values[1] if isinstance(shap_values_obj.base_values, list) else shap_values_obj.base_values
             print("  Using SHAP values for the positive class (output index 1).")
        else:
             # Handle cases where output might be single array or different structure
             shap_values_for_plot = shap_values_obj.values
             base_values_for_plot = shap_values_obj.base_values
             print("  Using single SHAP values output.")


        # Create SHAP Explanation object for plotting
        # Need to ensure data passed matches the SHAP values shape and has correct feature names
        shap_explanation = shap.Explanation(
             values=shap_values_for_plot,
             base_values=base_values_for_plot,
             data=X_sample, # Use the same sampled data
             feature_names=feature_names
         )


        print("  Generating summary plot...")
        plt.figure() # Ensure a new figure context
        shap.summary_plot(
            shap_explanation, # Use the Explanation object
            # X_sample, # Data is now inside Explanation object
            # feature_names=feature_names, # Names are inside Explanation object
            show=False, # Prevent immediate display
            max_display=min(num_features_to_show, X_sample.shape[1]) # Show top N or max available
        )
        plt.title("SHAP Summary Plot (Impact on model output)")
        plt.tight_layout()

        plt.savefig(output_file, bbox_inches='tight') # Save before showing
        print(f"✅ SHAP summary plot saved to {output_file}")

        if plt.get_backend().lower() != 'agg':
            plt.show(block=not no_wait)
            if no_wait:
                plt.pause(5)
        plt.close() # Close the figure

    except Exception as e:
        print(f"❌ Error generating SHAP plot: {e}")
        import traceback
        traceback.print_exc()


def ensure_unique_feature_names(feature_names):
    """Ensure that feature names are unique by appending suffixes if necessary."""
    if not feature_names:
        return []
    seen = {}
    unique_names = []
    for name in feature_names:
        original_name = name
        count = seen.get(name, 0)
        if count > 0: # Name already seen
             while name in seen:
                  suffix = count + 1 # Start suffix from _1, _2 etc.
                  name = f"{original_name}_{suffix}"
                  count = seen.get(name, 0) # Check if suffixed name also exists
                  if count == 0: # Found a unique suffixed name
                      break
                  # This case is unlikely but handles f_1, f_1_1 etc.
                  # Increment original count to try next suffix in outer loop if needed
                  seen[original_name] +=1
                  count = seen[original_name] # Get updated count for next suffix generation


        seen[name] = seen.get(name, 0) + 1 # Mark the (potentially suffixed) name as seen
        unique_names.append(name)

    # Optional: Check if any names were changed
    if unique_names != feature_names:
         print("⚠️ Duplicate feature names detected. Appended suffixes to ensure uniqueness.")
         # Example: print changes if needed for debugging
         # for old, new in zip(feature_names, unique_names):
         #      if old != new: print(f"   Renamed '{old}' -> '{new}'")

    return unique_names


def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM model and export to ONNX, supporting multiple input files.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs='+', # Accept one or more input arguments
        help="Input .json or .parquet feature file(s). Wildcards (e.g., data/*.json) are supported (quote if needed)."
    )
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX model path")
    parser.add_argument("--auto-tune", action="store_true", help="Perform hyperparameter tuning using GridSearchCV")
    parser.add_argument("--feature-names", type=str, default="names.txt", help="Path to a text file with feature names (one per line)")
    parser.add_argument("--no-wait", action="store_true", help="Show plots without blocking execution (plots close after a short pause)")
    args = parser.parse_args()

    # --- Process Input Files ---
    input_files = []
    print("🔍 Resolving input file paths...")
    for pattern in args.input:
        # Use glob to expand wildcards
        expanded_files = glob.glob(pattern)
        if not expanded_files:
            print(f"⚠️ Warning: No files matched the pattern: {pattern}")
        else:
            input_files.extend(expanded_files)

    if not input_files:
        print("❌ Error: No valid input files found after expanding patterns.")
        return # Exit if no files

    input_files = sorted(list(set(input_files))) # Get unique sorted list
    print(f"🔢 Found {len(input_files)} unique input file(s) to process:")
    for f in input_files:
        print(f"  - {f}")


    # --- Load Feature Names ---
    feature_names = None
    if args.feature_names:
        try:
            with open(args.feature_names, 'r') as f:
                # Read names, strip whitespace, and remove empty lines
                feature_names = [line.strip() for line in f if line.strip()]
            if not feature_names:
                 print(f"⚠️ Feature names file '{args.feature_names}' is empty or contains only whitespace.")
                 feature_names = None # Treat as if no file provided
            else:
                 print(f"📝 Loaded {len(feature_names)} feature names from '{args.feature_names}'.")
                 # Ensure uniqueness right after loading
                 feature_names = ensure_unique_feature_names(feature_names)
        except FileNotFoundError:
            print(f"⚠️ Feature names file not found: {args.feature_names}. Feature names will be auto-generated if needed.")
            feature_names = None
        except Exception as e:
             print(f"❌ Error reading feature names file '{args.feature_names}': {e}")
             return # Stop execution if feature name loading fails


    # --- Load Data ---
    try:
        # Pass the list of resolved files and potentially loaded feature names
        X, y, loaded_feature_names = load_data(input_files, feature_names)
        # Update feature_names if they were generated inside load_data
        if feature_names is None and loaded_feature_names:
             feature_names = loaded_feature_names
             print("✅ Using feature names generated during data loading.")

    except (FileNotFoundError, ValueError, Exception) as e:
         print(f"❌ Failed to load data: {e}")
         return # Exit if data loading fails


    # --- Hyperparameter Tuning Setup ---
    # Define hyperparameters for tuning (only used if --auto-tune is specified)
    param_grid = None
    if args.auto_tune:
        print("📋 Setting up hyperparameter grid for tuning...")
        param_grid = {
            'num_leaves': [31, 50, 70],        # Smaller values for potentially smaller datasets
            'max_depth': [-1, 10, 15],         # -1 means no limit
            'learning_rate': [0.05, 0.1],    # Common learning rates
            'n_estimators': [100, 200],      # Number of trees
            # Add other parameters like reg_alpha, reg_lambda, colsample_bytree if needed
            # 'reg_alpha': [0.0, 0.1], # L1 regularization
            # 'reg_lambda': [0.0, 0.1], # L2 regularization
            # 'colsample_bytree': [0.8, 1.0], # Feature fraction
        }


    # --- Train and Export ---
    try:
        train_and_export_onnx(
            X,
            y,
            args.output,
            feature_names, # Pass the final list of feature names
            hyperparameters=param_grid, # Pass grid if tuning, else None
            no_wait=args.no_wait
        )
        print("\n🎉 Script finished successfully!")
    except Exception as e:
        print(f"\n❌ An error occurred during training or export: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        print("\n Script finished with errors.")


if __name__ == "__main__":
    main()