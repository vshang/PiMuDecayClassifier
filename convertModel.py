import json
import os

def generate_scaler_code(scaler_params, header, source, output_basename):
    means = scaler_params["mean"]
    scales = scaler_params["scale"]
    n_features = len(means)

    header.write("#pragma once\n\n")
    header.write(f"const int N_FEATURES = {n_features};\n")
    header.write("extern const double mean[N_FEATURES];\n")
    header.write("extern const double scale[N_FEATURES];\n")
    header.write("void scale_input(double* x);\n")
    header.write("double predict(const double* x);\n")

    source.write(f"#include \"{output_basename}.h\"\n")
    source.write("#include <cmath>\n\n")

    source.write("const double mean[N_FEATURES] = {" + ", ".join(map(str, means)) + "};\n")
    source.write("const double scale[N_FEATURES] = {" + ", ".join(map(str, scales)) + "};\n\n")

    source.write("void scale_input(double* x) {\n")
    source.write("    for (int i = 0; i < N_FEATURES; ++i)\n")
    source.write("        x[i] = (x[i] - mean[i]) / scale[i];\n")
    source.write("}\n\n")

def generate_tree_function(tree, tree_id):
    def recurse(node):
        if "leaf" in node:
            return f"return {node['leaf']};\n"
        elif "split" in node and "split_condition" in node and "children" in node:
            fid = int(node["split"][1:])
            threshold = node["split_condition"]
            children = node["children"]

            if len(children) != 2:
                raise ValueError(
                    f"Tree node does not have exactly 2 children. Found {len(children)}."
                )

            code = f"if (x[{fid}] < {threshold}) {{\n"
            code += recurse(children[0])
            code += "} else {\n"
            code += recurse(children[1])
            code += "}\n"
            return code
        else:
            raise ValueError(
                f"Unexpected node structure. Node keys: {list(node.keys())}"
            )

    func = f"double tree_{tree_id}(const double* x) {{\n"
    func += recurse(tree)
    func += "}\n\n"
    return func

def generate_tree_code(xgb_model, header, source):
    # In dump_model format, the JSON is a list of trees directly
    trees = xgb_model
    n_trees = len(trees)

    source.write("// Tree functions\n")
    for i, tree in enumerate(trees):
        tree_func = generate_tree_function(tree, i)
        source.write(tree_func)

    # Aggregate predictions and convert to probability
    source.write("double predict(const double* x) {\n")
    source.write("    double score = 0.0;\n")
    for i in range(n_trees):
        source.write(f"    score += tree_{i}(x);\n")
    source.write("    double proba = 1.0 / (1.0 + std::exp(-score));\n")
    source.write("    return proba;\n")
    source.write("}\n")

def generate_cpp_code(input_folder="exported_model", output_folder="generated_code", output_basename="model"):
    scaler_file = os.path.join(input_folder, "scaler_params.json")
    model_file = os.path.join(input_folder, "xgb_model.json")

    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler parameters file not found: {scaler_file}")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"XGBoost model file not found: {model_file}")

    os.makedirs(output_folder, exist_ok=True)

    with open(scaler_file) as f:
        scaler_params = json.load(f)

    with open(model_file) as f:
        xgb_model = json.load(f)

    header_path = os.path.join(output_folder, f"{output_basename}.h")
    source_path = os.path.join(output_folder, f"{output_basename}.cc")

    with open(header_path, "w") as header, open(source_path, "w") as source:
        generate_scaler_code(scaler_params, header, source, output_basename)
        generate_tree_code(xgb_model, header, source)

    print(f" C++ files successfully written to: {output_folder}/{output_basename}.h and .cc")