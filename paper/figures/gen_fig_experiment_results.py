"""Publication charts from archived aggregates; no inference or new statistics."""
from pathlib import Path
import csv
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PUB = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/02_完整统计"
MECH = ROOT / "runs/mechanism_eval/qzbc_care_opd_v1"
BLUE, GREEN, ORANGE, GRAY = "#0072B2", "#009E73", "#D55E00", "#8D99A6"
INK = "#243345"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "legend.frameon": False,
})

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save(fig, name, paths, data):
    fig.canvas.draw()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", pad_inches=.03)
    manifest = {"source_files": [{"path": str(p.relative_to(ROOT)),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                "data": data, "statistics": "Archived values; no smoothing or resampling."}
    (HERE / f"{name}.data.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plt.close(fig)

def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E9EE", linewidth=.6)
    ax.tick_params(length=3, pad=3)

def agent():
    toolpath = PUB / "AGENT_TOOL_CONTRACT_FULL.csv"
    matrixpath = PUB / "AGENT_SYSTEM_MATRIX.csv"
    tool = {r["endpoint"]: r for r in rows(toolpath)}
    matrix = rows(matrixpath)
    fig = plt.figure(figsize=(7.16, 2.65), facecolor="white")
    a = fig.add_axes([.14, .19, .32, .62])
    b = fig.add_axes([.62, .19, .35, .62])
    fig.text(.04, .95, "(a) Authorized tool execution", weight="bold", size=9)
    fig.text(.54, .95, "(b) Model and orchestration", weight="bold", size=9)
    endpoints = ["tool_exact", "tool_micro_precision", "tool_micro_recall", "task_success", "evidence_source_coverage"]
    names = ["Tool Exact", "Precision", "Recall", "Task success", "Source coverage"]
    y = np.arange(5)[::-1]
    v0 = [100*float(tool[k]["legacy_role_wide"]) for k in endpoints]
    v1 = [100*float(tool[k]["minimal_tool"]) for k in endpoints]
    a.hlines(y, v0, v1, color="#B6C2CC", lw=1.5)
    a.scatter(v0, y, c=BLUE, marker="o", s=26, label="Role-wide", zorder=3)
    a.scatter(v1, y, c=ORANGE, marker="s", s=26, label="MinimalTool", zorder=4)
    a.set(yticks=y, yticklabels=names, xlim=(0, 105), ylim=(-.5, 4.6), xlabel="Rate (%)")
    a.set_xticks([0,25,50,75,100]); grid(a)
    a.legend(loc="lower left", bbox_to_anchor=(-.04, 1.0), ncol=2, handletextpad=.4, columnspacing=1.0)
    arches = ["Direct", "RAG", "Guard Agent", "ReAct"]
    vv = []
    for i, arch in enumerate(arches):
        pair = [100*float(next(r for r in matrix if r["inference_architecture"]==arch and r["model_state"]==state)["mcq_accuracy"])
                for state in ["base", "CARE-OPD post-trained"]]
        yy = 3-i
        b.plot(pair, [yy, yy], color="#B6C2CC", lw=1.5)
        for j, (value, col, marker) in enumerate(zip(pair, [BLUE, ORANGE], ["o", "s"])):
            b.scatter(value, yy, c=col, marker=marker, s=26, zorder=3)
            b.annotate(f"{value:.2f}", (value, yy), xytext=(0, -12 if j==0 else 6),
                       textcoords="offset points", ha="center", fontsize=7, color=col)
        vv.append(pair)
    b.set(yticks=[3,2,1,0], yticklabels=["Direct", "RAG", "Guard", "ReAct"],
          xlim=(65.5,78), ylim=(-.6,3.6), xlabel="MCQ accuracy (%)")
    b.set_xticks([66,69,72,75,78]); grid(b)
    b.scatter([], [], c=BLUE, marker="o", label="Base")
    b.scatter([], [], c=ORANGE, marker="s", label="CARE-OPD-trained")
    b.legend(loc="lower left", bbox_to_anchor=(-.02, 1.0), ncol=2, handletextpad=.4, columnspacing=.9)
    save(fig, "fig_agent_execution", [toolpath, matrixpath], {"endpoints": endpoints, "role_wide":v0,"minimal_tool":v1,"architecture":arches,"mcq_pairs":vv})

def mechanism():
    transpath = PUB / "QZBC_STATE_AND_TRANSITION_FULL.csv"
    roundpath = MECH / "round_dynamics/round_dynamics.csv"
    abpath = ROOT / "summary_vs_without_retention/summary.csv"
    abr = {r["model"]: float(r["perturbed_tsr"])*100 for r in rows(abpath)}
    trans = {r["model"]: r for r in rows(transpath) if r["record_type"]=="transition"}
    rs = rows(roundpath)
    fig = plt.figure(figsize=(7.16, 3.1), facecolor="white")
    a = fig.add_axes([.12, .22, .20, .59])
    b = fig.add_axes([.51, .22, .20, .59])
    c = fig.add_axes([.80, .22, .18, .59])
    fig.text(.025, .96, "(a) Recovery and regression", weight="bold", size=9)
    fig.text(.385, .96, "(b) Matched effects", weight="bold", size=9)
    fig.text(.765, .96, "(c) Current failures", weight="bold", size=9)
    models=["Frozen","Recovery-SFT","Recovery-DPO","GKD-Full","CARE-OPD"]
    yy=np.arange(5)[::-1]
    rescued=[int(trans[k]["rescued"]) for k in models]
    regressed=[int(trans[k]["regressed"]) for k in models]
    a.barh(yy, -np.array(regressed), height=.58, color=ORANGE, label="Regressed")
    a.barh(yy, rescued, height=.58, color=GREEN, label="Rescued")
    for y, rr, gg in zip(yy, rescued, regressed):
        a.text(rr+45,y,str(rr),va="center",ha="left",size=7)
        a.text(-gg-45,y,str(gg),va="center",ha="right",size=7)
    a.axvline(0,color=INK,lw=.7)
    a.set(yticks=yy,yticklabels=["Frozen","Recovery-SFT","Recovery-DPO","GKD-Full","CARE-OPD"],xlim=(-2900,1700),xlabel="Changed requests",ylim=(-.6,4.6))
    a.set_xticks([-2000,0,1000]); a.set_xticklabels(["−2000","0","1000"])
    grid(a); a.legend(loc="upper left", bbox_to_anchor=(-.38,1.18),ncol=2,handlelength=1,columnspacing=.7)
    comparisons=[("Selected − full", "without_retention","matched_full_response",8.92,12.31),
                 ("Selected − outside","without_retention","outside_only",4.10,5.64),
                 ("With − w/o Protect","full_selected","without_retention",1.88,4.92),
                 ("With − w/o trust","full_selected","without_trust",1.40,4.00),
                 ("With − w/o risk","full_selected","without_risk",-3.62,-.48)]
    points=[]
    for y,(label,left,right,lo,hi) in zip(yy,comparisons):
        d=abr[left]-abr[right];points.append({"label":label,"difference":d,"ci95":[lo,hi]})
        col=BLUE if d>0 else ORANGE
        b.errorbar(d,y,xerr=[[d-lo],[hi-d]],fmt="o",color=col,capsize=3,ms=4,lw=1.3)
        b.text(d,y+.24,f"{d:+.2f}",ha="center",size=7,color=col)
    b.axvline(0,color=INK,lw=.7,ls="--");grid(b)
    b.set(yticks=yy,yticklabels=[r[0] for r in comparisons],xlim=(-5,14),ylim=(-.6,4.6),xlabel="Perturbed TSR gain (pp)")
    b.set_xticks([-4,0,4,8,12])
    b.text(.5,1.08,"Paired 95% bootstrap CI",transform=b.transAxes,ha="center",size=7.5)
    rounds=[r for r in rs if r["row_type"]=="round"]
    transitions=[r for r in rs if r["row_type"]=="transition" and r["failure_class"]=="all"]
    previous=[0]+[int(r["persistent"]) for r in transitions]
    new=[int(rounds[0]["failed"])]+[int(r["new"]) for r in transitions]
    xs=[1,2,3]
    c.bar(xs,[new[0],0,0],width=.62,color=GRAY,label="Initial")
    c.bar(xs,[0,new[1],new[2]],width=.62,color=BLUE,label="New")
    c.bar(xs,previous,bottom=new,width=.62,color=GREEN,label="Persistent")
    for x,n,p in zip(xs,new,previous):
        c.text(x,n+p+75,str(n+p),ha="center",size=7.5)
    c.set(xlabel="Collection round",ylabel="Failed rollouts",xticks=xs,ylim=(0,4150),yticks=[0,1000,2000,3000,4000])
    c.set_axisbelow(True);c.grid(axis="y",color="#E4E9EE",lw=.6)
    c.legend(loc="upper left",bbox_to_anchor=(-.20,-.24),ncol=2,handlelength=1,columnspacing=.6)
    save(fig,"fig_behavior_mechanism",[transpath,roundpath,abpath,ROOT/"summary_vs_full_selected/summary.json",ROOT/"summary_vs_without_retention/summary.json"],
         {"models":models,"rescued":rescued,"regressed":regressed,"matched_effects":points,"initial_or_new":new,"persistent":previous,
          "ci_note":"Archived paired intervals, not recalculated. Localization contrasts disable Protect in both arms."})

if __name__=="__main__":
    agent()
    mechanism()
    print("Generated fig_agent_execution and fig_behavior_mechanism with source manifests.")
