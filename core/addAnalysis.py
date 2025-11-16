from sklearn.ensemble import RandomForestClassifier
import pickle
from collections import defaultdict

def _process_rules_for_prompt(model_out_list):
    """
    Helper function to process the rule list and combine by feature.
    Example input: [('product_B_sold_in_the_past <= -0.34', 105), ...]
    """
    
    grouped_rules = defaultdict(list)
    
    for rule, count in model_out_list:
        try:
            # Assumes rule format: 'feature_name operator value'
            parts = rule.split(' ')
            feature = parts[0]
            condition = " ".join(parts[1:])
            # Add a business-friendly explanation of the count
            grouped_rules[feature].append(f"{condition} (this rule was a factor in {count} similar cases)")
        except Exception:
            # Fallback for any unexpected rule format
            grouped_rules["Other rules"].append(f"{rule} (a factor in {count} similar cases)")

    # Format this dictionary into a clean string for the prompt
    prompt_string = ""
    for feature, conditions in grouped_rules.items():
        prompt_string += f"\n- For '{feature}':\n"
        for cond_str in conditions:
            prompt_string += f"    - {cond_str}"
    
    return prompt_string

def top7_model(instance):
    with open('model/objs.pkl', 'rb') as f:
        [X_train, y_train, X_test, y_test, all_features] = pickle.load(f)
    top_7 = all_features[:7]
    # len(top_7)
    X_tr = X_train[top_7]
    X_te = X_test[top_7]

    rf_c = RandomForestClassifier(n_estimators=65,
        max_depth=32,
        random_state=42,
        min_samples_split=6,
        min_samples_leaf=1,
        n_jobs=-1)
    rf_c.fit(X_tr, y_train)

    model = rf_c
    feature_names = top_7
    instance = instance[:7]

    rf_c.predict(instance.reshape(1, -1))

    # Assume 'model' is your fitted RandomForestClassifier
    # Assume 'instance' is your 1D numpy array of a single data point
    # Assume 'feature_names' is a list of your feature names

    all_paths_rules = []

    # 1. Iterate through each tree
    for tree in model.estimators_:
        tree_rules = []
        node_index = 0 # Start at the root node
        
        # 2. Traverse down the tree until a leaf is reached
        while tree.tree_.feature[node_index] != -2: # -2 means it's a leaf
            
            # Get feature and threshold for the current node
            feature_idx = tree.tree_.feature[node_index]
            feature_name = feature_names[feature_idx]
            threshold = tree.tree_.threshold[node_index]
            
            # 3. Decide which path to take
            if instance[feature_idx] <= threshold:
                # Record the rule
                tree_rules.append(f"{feature_name} <= {threshold:.2f}")
                # Move to the left child
                node_index = tree.tree_.children_left[node_index]
            else:
                # Record the rule
                tree_rules.append(f"{feature_name} > {threshold:.2f}")
                # Move to the right child
                node_index = tree.tree_.children_right[node_index]
                
        # 4. Once at a leaf, get the prediction for this tree
        # tree_prediction = tree.predict(instance.reshape(1, -1))
        # You can also get class probabilities from the leaf node's 'value'
        # leaf_values = tree.tree_.value[node_index]
        
        # 5. Store the set of rules for this tree's path
        all_paths_rules.append(tree_rules)

    # 6. Aggregate the results
    # You can now analyze 'all_paths_rules'
    # For example, count the frequency of each individual rule
    from collections import Counter
    all_rules_flat = [rule for path in all_paths_rules for rule in path]
    rule_counts = Counter(all_rules_flat)

    return _process_rules_for_prompt(rule_counts.most_common(10))  
    