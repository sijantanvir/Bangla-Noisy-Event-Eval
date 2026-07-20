from seqeval.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_preds, id2label):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        cp, cl = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                cp.append(id2label[int(p)])
                cl.append(id2label[int(l)])
        true_preds.append(cp)
        true_labels.append(cl)

    return {
        "f1": f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "macro_f1": f1_score(true_labels, true_preds, average="macro"),
    }
