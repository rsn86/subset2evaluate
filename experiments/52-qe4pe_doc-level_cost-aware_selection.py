# %%
import collections
import subset2evaluate.utils as utils
import subset2evaluate.evaluate
import subset2evaluate.select_subset
import numpy as np

data_old_all = list(utils.load_data_qe4pe(cache_fname='gsarti_qe4pe_main').values())
spa_all_random = []

# %%
# Cost-aware selection
# Ref: https://github.com/zouharvi/subset2evaluate?tab=readme-ov-file#advanced-usage
data_full = data_old_all[0]  # Use the first dataset as the full data
# print("Cost-aware selection")
# print(data_full[0]['cost'])

# run basic selection
# The only non aggregated segment level metrics available are pe_xcomet_qe and qa_pe_esa_rating
data_new = subset2evaluate.select_subset.basic(data_full, method="metric_avg", metric="pe_xcomet_qe")

cost_first_23 = sum([line["cost"] for line in data_new[:23]])
print(f"Cost for the first 23 segments: {cost_first_23:.2f}")

corr_first_23 = subset2evaluate.evaluate.eval_subset_correlation(data_new[:23], data_full, metric="pe_xcomet_qe")
print(f"Correlation for the first 23 segments: {corr_first_23:.2f}")

print("-"* 80)
# let's run cost-aware selection
data_costaware = subset2evaluate.select_subset.costaware(data_new, budget=cost_first_23, metric="pe_xcomet_qe")

cost_costaware = sum([line["cost"] for line in data_costaware])
print(f"Cost of the cost-aware selection: {cost_costaware:.2f} #{len(data_costaware)} segsments")

corr_costaware = subset2evaluate.evaluate.eval_subset_correlation(data_costaware, data_full, metric="pe_xcomet_qe")
print(f"Correlation of the cost-aware selection: {corr_costaware:.2f}")

# %%


def aggregate_doc_score(doc, agg_method=np.average, metric="human"):
    sys_scores = collections.defaultdict(list)
    """Aggregate the scores of a document."""
    for line in doc:
        for sys,scores in line['scores'].items():
            if metric not in scores:
                raise ValueError(f"ERROR: {metric} not found in {sys} scores for {line['i']}")
            sys_scores[sys].append(scores[metric])

    return {sys: {metric: agg_method(scores)} for sys, scores in sys_scores.items()}



def aggregate_doc(data_scored, agg_method=np.sum, metric="human"):
    """Aggregate the scored data by document."""
    data_aggregated = collections.defaultdict(list)
    for line in data_scored:
        data_aggregated[line["doc"]].append(line)

    data_aggregated = [
        {
            "doc": doc,
            "i": lines[0]["i"],  # keep just the first index for the document
            # sum the utilities across the document
            "subset2evaluate_utility": agg_method([line["subset2evaluate_utility"] for line in lines]),
            # sum the cost across the document
            "cost": agg_method([line["cost"] for line in lines]),
            "scores": aggregate_doc_score(lines, agg_method=np.average, metric=metric),
        }
        for doc, lines in data_aggregated.items()
    ]
    data_aggregated.sort(key=lambda x: x["subset2evaluate_utility"], reverse=True)
    return data_aggregated

print("-"* 80)
data_aggregated = aggregate_doc(data_new, metric="pe_xcomet_qe")

# let's run cost-aware selection
data_aggregated_costaware = subset2evaluate.select_subset.costaware(data_aggregated, budget=cost_first_23, metric="pe_xcomet_qe")

cost_aggregated_costaware = sum([line["cost"] for line in data_aggregated_costaware])
print(f"Cost of the doc aggregated cost-aware selection: {cost_aggregated_costaware:.2f} #{len(data_aggregated_costaware)} documents")

corr_aggregated_costaware = subset2evaluate.evaluate.eval_subset_correlation(data_aggregated_costaware, data_full, metric="pe_xcomet_qe")
print(f"Correlation of the doc aggregated cost-aware selection: {corr_aggregated_costaware:.2f}")
