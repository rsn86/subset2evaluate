# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np
import tqdm

data_old_all = list(utils.load_data_qe4pe(cache_fname='gsarti_qe4pe_main').values())
spa_all_random = []

# %%
# Cost-aware selection
# Ref: https://github.com/zouharvi/subset2evaluate?tab=readme-ov-file#advanced-usage
data_full = data_old_all[0]  # Use the first dataset as the full data
print("Cost-aware selection")
print(data_full[0]['cost'])

# run basic selection
# The only non aggregated segment level metrics available are pe_xcomet_qe and qa_pe_esa_rating
data_new = subset2evaluate.select_subset.basic(data_full, method="metric_var", metric="pe_xcomet_qe")

cost_first_23 = sum([line["cost"] for line in data_new[:23]])
print(f"Cost for the first 23 items: {cost_first_23:.2f}")

corr_first_23 = subset2evaluate.evaluate.eval_subset_correlation(data_new[:23], data_full, metric="pe_xcomet_qe")
print(f"Correlation for the first 23 items: {corr_first_23:.2f}")

# let's run cost-aware selection
data_costaware = subset2evaluate.select_subset.costaware(data_new, budget=cost_first_23, metric="pe_xcomet_qe")

cost_costaware = sum([line["cost"] for line in data_costaware])
print(f"Cost of the cost-aware selection: {cost_costaware:.2f}")

corr_costaware = subset2evaluate.evaluate.eval_subset_correlation(data_costaware, data_full, metric="pe_xcomet_qe")
print(f"Correlation of the cost-aware selection: {corr_costaware:.2f}")

# %%

# experiments/24-document_level.py
for repetitions, method_kwargs in [
    (10, dict(method="random")),
    # (1, dict(method="metric_avg", metric="MetricX-23-c")),
    # (1, dict(method="metric_var", metric="MetricX-23-c")),
    # (1, dict(method="metric_cons", metric="MetricX-23-c")),
    # (1, dict(method="diversity", metric="lm")),
    # (5, dict(method="pyirt_diffdisc", metric="MetricX-23-c", model="4pl_score", retry_on_error=True)),
    # (1, dict(method="precomet_avg")),
    # (1, dict(method="precomet_var")),
    # (1, dict(method="precomet_cons")),
    # (1, dict(method="precomet_diversity")),
    # (1, dict(method="precomet_diffdisc_direct")),
]:
    load_model = None
    spa_all = []
    for data_old in tqdm.tqdm(data_old_all):
        for _ in range(repetitions):
            def evaluate_aggregate_doc(data_scored):
                data_old_aggregated = collections.defaultdict(list)
                for line in data_scored:
                    data_old_aggregated[line["doc"]].append(line)

                data_old_aggregated = [
                    {
                        "doc": doc,
                        "i": [line["i"] for line in lines],
                        # average the utilities across the document
                        "subset2evaluate_utility": np.average([line["subset2evaluate_utility"] for line in lines]),
                        # average the cost across the document
                        "cost": np.average([line["cost"] for line in lines])
                    }
                    for doc, lines in data_old_aggregated.items()
                ]
                data_old_aggregated.sort(key=lambda x: x["subset2evaluate_utility"], reverse=True)
                data_new_flat = [
                    data_old[i]
                    for doc in data_old_aggregated
                    for i in doc["i"]
                ]
                return subset2evaluate.evaluate.eval_spa(data_new_flat, data_old, metric="pe_xcomet_qe")

            data_y, load_model = subset2evaluate.select_subset.basic(
                data_old,
                **method_kwargs,
                load_model=load_model if method_kwargs["method"] != "pyirt_diffdisc" else None,
                return_model=True
            )
            spa_all.append(evaluate_aggregate_doc(data_y))
            if method_kwargs["method"] == "random":
                spa_all_random.append(spa_all[-1])

    print(method_kwargs["method"], f"{np.average(spa_all):.1%}")


# %%

import subset2evaluate.utils

spa_all_random_arr = np.array(spa_all_random).mean(axis=1).reshape(repetitions, -1).mean(axis=0)
conf = subset2evaluate.utils.confidence_interval(spa_all_random_arr, confidence=0.90,)
print(f"{(conf[1]-conf[0])/2:.2%}")

# %%

### TODO: Document-level cost-aware selection
