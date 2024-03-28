import re

AUROC_REGEXES = [
    r"\bAUC?\-?\(?ROC\)?\b",
    r"\bAUC\b",
    r"\barea under the curve\b",
    r"\bROC\b",
    r"\breceiver operating characteristic\b",
    r"sensitivity \s*(vs\.?|v\.?|versus|against|compared with) \s*(1\s?-\s?specificity|specificity)",
    r"(true positive rate|TPR) \s*(vs\.?|v\.?|versus|against|compared with) \s*(false positive rate|FPR)"
]

AUPRC_REGEXES = [
    r"\bAUC?\-?\(?PRC\)?\b",
    r"\bprecision[\s-]?recall\b", 
    r"\bAPR\b",
    r"\baverage[\s-]?precision\b",
    r"\bPRC\b"
]

COMBINED_AUROC_REGEX = r"(?i)(" + '|'.join(AUROC_REGEXES) + r")"
compiled_auroc_regex = re.compile(COMBINED_AUROC_REGEX)

COMBINED_AUPRC_REGEX = r"(?i)(" + '|'.join(AUPRC_REGEXES) + r")"
compiled_auprc_regex = re.compile(COMBINED_AUPRC_REGEX)
