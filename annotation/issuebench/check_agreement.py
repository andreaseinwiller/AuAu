from llm_audit.eval.issuebench import agreement, load_issuebench_annotations

# paths_andreas = ["resources/input/datasets/issuebench/annotations_rebuttal_andreas.json"]
# paths_andreas = ["resources/input/datasets/issuebench/annotations_rebuttal_no-conflicts.json"]
# paths_max = ["resources/input/datasets/issuebench/annotations_rebuttal_no-conflicts.json"]
# paths_max = ["resources/input/datasets/issuebench/annotations_rebuttal_max.json"]
paths_max = ["resources/input/datasets/issuebench/annotations_rebuttal_non-autho_max.json"]
paths_andreas = ["resources/input/datasets/issuebench/annotations_rebuttal_non-autho_andreas.json"]

agree, descriptive = agreement(paths_andreas, paths_max)

print(agree.to_markdown())
print(descriptive.to_markdown())
print(load_issuebench_annotations(paths_max).columns)
