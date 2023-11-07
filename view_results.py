import os
import pandas as pd
import json

inspect_case = "umdfaces32"

RESULTS_DIR = "/p/distinf/Face-Auditor/WORKDIR/temp_data/results/"
all_data = []
for path in os.listdir(RESULTS_DIR):
    if path.startswith(inspect_case) and path.endswith("_notmean.json"):
        with open(os.path.join(RESULTS_DIR, path), 'r') as f:
            data = json.load(f)
            case = path.split("_")[1].split(".")[0]
            all_data.append({
                "experiment": case,
                "model_train_acc": '%.3f±%.3f' % (data["target_train_acc"], data["target_train_acc_std"]),
                "model_test_acc": '%.3f±%.3f' % (data["target_test_acc"], data["target_test_acc_std"]),
                "attack_acc": '%.3f±%.3f' % (data["attack_acc"], data["attack_acc_std"]),
                "attack_auc": '%.3f±%.3f' % (data["attack_auc"], data["attack_auc_std"]),
                "attack_F1": '%.3f±%.3f' % (data["attack_f1_score"], data["attack_f1_score_std"]),
            })

# Make pandas dataframe
df = pd.DataFrame(all_data)
# Print (display all rows)
print(df)
