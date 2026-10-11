"""Write the LaTeX tables used in the report to report/tables/.

  python experiments/make_tables.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, OUT = f"{ROOT}/results", f"{ROOT}/report/tables"
NAMES = {"cifar10": "CIFAR-10", "catsdogs": "Cats vs Dogs"}


def pct(x):
    return f"{100 * x:.1f}"


def k(n):
    return f"{n / 1000:.0f}k" if n >= 10000 else f"{n / 1000:.1f}k"


def e1():
    rows = [json.loads(l) for l in open(f"{RES}/e1_overfit.jsonl")]
    order = {"random": 0, "cnn": 1, "resnet": 2}
    rows.sort(key=lambda r: (r["dataset"] != "cifar10", order[r["trunk"]], r["mode"] != "shared"))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"dataset & features & params & per image & epochs & train acc & test acc & test acc (hard) \\",
             r"\midrule"]
    prev = None
    for r in rows:
        if prev and prev != r["dataset"]:
            lines.append(r"\midrule")
        prev = r["dataset"]
        lines.append(f"{NAMES[r['dataset']]} & {r['trunk']}, {r['mode']} & {k(r['params'])} & {k(r['active_params'])} & "
                     f"{r['epochs']} & {pct(r['train_acc'])} & {pct(r['test_acc'])} & {pct(r['hard_test_acc'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e3(dataset, trunk):
    d = json.load(open(f"{RES}/e3_{dataset}_{trunk}.json"))
    lines = [r"\begin{tabular}{rrlllr}", r"\toprule",
             r"node & images & left: top classes & right: top classes & NMI(split, class) & saliency radius \\",
             r"\midrule"]
    for r in d["routers"]:
        lines.append(f"{r['node']} & {r['n_reach']} & {', '.join(r['left_top'][:2])} & {', '.join(r['right_top'][:2])} & "
                     f"{r['nmi_split_class']:.3f} & {r['saliency_radius']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e3_summary():
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"dataset & trunk & NMI(leaf, class) & NMI(root, vehicle) & mean router NMI & router radius & leaf radius \\",
             r"\midrule"]
    for ds in ("cifar10", "catsdogs"):
        for trunk in ("resnet", "cnn"):
            d = json.load(open(f"{RES}/e3_{ds}_{trunk}.json"))
            rr = [r["saliency_radius"] for r in d["routers"]]
            lr = [r["saliency_radius"] for r in d["leaves"] if "saliency_radius" in r]
            rn = sum(r["nmi_split_class"] for r in d["routers"]) / len(d["routers"])
            veh = f"{d['nmi_root_vehicle']:.3f}" if "nmi_root_vehicle" in d else "--"
            lines.append(f"{NAMES[ds]} & {trunk} & {d['nmi_leaf_class']:.3f} & {veh} & {rn:.3f} & "
                         f"{min(rr):.2f}--{max(rr):.2f} & {min(lr):.2f}--{max(lr):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e4():
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{CIFAR-10} & & \multicolumn{2}{c}{Cats vs Dogs} \\",
             r"\cmidrule{2-3}\cmidrule{5-6}",
             r"model & params & test acc & & params & test acc \\", r"\midrule"]
    ds = {n: json.load(open(f"{RES}/e4_{n}.json")) for n in ("cifar10", "catsdogs")}

    def row(label, f):
        cells = []
        for n in ("cifar10", "catsdogs"):
            p, a = f(ds[n])
            cells.append(f"{p} & {pct(a)}")
        lines.append(f"{label} & {cells[0]} & & {cells[1]} \\\\")

    row("E1 tree, soft routing", lambda d: (k(d["base"]["params"]), d["base"]["test_acc_soft"]))
    row("E1 tree, hard routing (per image)", lambda d: (k(d["base"]["active_params"]), d["base"]["test_acc_hard"]))
    row("int8 weights (same count, 4x fewer bytes)", lambda d: (k(d["base"]["params"]), d["int8"]["test_acc"]))
    for s in (0.9, 0.99):
        row(f"magnitude pruned {int(s * 100)}\\%, stored",
            lambda d, s=s: next((k(r["nonzero"]), r["test_acc"]) for r in d["magnitude"] if r["sparsity"] == s))
        row(f"magnitude pruned {int(s * 100)}\\%, per image",
            lambda d, s=s: next((k(r["active_nonzero"]), r["test_acc"]) for r in d["magnitude"] if r["sparsity"] == s))
    row("subtree collapse, at most 3 leaves",
        lambda d: next((f"{k(r['params'])} ({r['leaves']} leaves)", r["test_acc"]) for r in d["collapse"] if r["leaves"] <= 3))
    row("linear classifier, trained on labels",
        lambda d: next((k(r["params"]), r["test_acc_hard"]) for r in d["small"] if r["depth"] == 0 and r["mode"] == "labels"))
    row("linear classifier, distilled from E1 tree",
        lambda d: next((k(r["params"]), r["test_acc_hard"]) for r in d["small"] if r["depth"] == 0 and r["mode"] == "distill"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def e5():
    rows = json.load(open(f"{RES}/e5_cifar10.json"))
    lines = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
             r"depth & width & params & test err & copies noisy NN label & err, NN noisy & err, NN clean & label-routed (control) \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['depth']} & {r['h']} & {k(r['params'])} & {pct(r['test_err'])} & "
                     f"{pct(r['p_pred_eq_noisy_nn_label'])} & {pct(r['test_err_nn_noisy'])} & {pct(r['test_err_nn_clean'])} & "
                     f"{pct(r['label_routed'])} ({pct(r['label_routed_control'])}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = {"e1.tex": e1, "e4.tex": e4, "e5.tex": e5, "e3_summary.tex": e3_summary,
            "e3_cifar10.tex": lambda: e3("cifar10", "resnet"), "e3_catsdogs.tex": lambda: e3("catsdogs", "resnet")}
    for name, fn in jobs.items():
        try:
            text = fn()
        except FileNotFoundError as e:
            print("skip", name, e)
            continue
        open(f"{OUT}/{name}", "w").write(text + "\n")
        print("wrote", name)
