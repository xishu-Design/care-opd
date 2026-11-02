"""Real-record state profiles and isolated-generator latency from frozen CSVs."""
from gen_fig_experiment_results import *

def main():
    source = ROOT / "competition_project_document/formal/competition_submission_v9/supporting_public_v9/03_真实场景外测/统一统计合同/MODEL_TRACK_STATE_METRICS.csv"
    deploy = PUB / "DEPLOYMENT_AUDIT_FULL.csv"
    data = rows(source)
    models = ["frozen", "clean_sft", "recovery_sft", "recovery_dpo", "gkd_full", "care_opd"]
    labels = ["Frozen", "Clean-SFT", "Recovery-SFT", "Recovery-DPO", "GKD-Full", "CARE-OPD"]
    variants = ["clean", "bad_history", "source_outage", "evidence_conflict"]
    tracks = ["structured_evidence_faithfulness", "raw_strict_extraction_challenge"]
    matrices = [np.array([[100*float(next(r for r in data if r["model"]==m and r["track"]==t and r["variant"]==v)["task_success_rate"])
                          for v in variants] for m in models]) for t in tracks]
    fig=plt.figure(figsize=(7.16,2.85),facecolor="white")
    axes=[fig.add_axes([.14,.23,.25,.60]),fig.add_axes([.43,.23,.25,.60])]
    titles=["(a) Structured (65 families)","(b) Raw strict (54 families)"]
    for i,(ax,values,title) in enumerate(zip(axes,matrices,titles)):
        im=ax.imshow(values,vmin=0,vmax=100,cmap="Blues",aspect="auto")
        ax.set_xticks(range(4),["Clean","History","Outage","Conflict"],fontsize=7)
        ax.set_yticks(range(6),labels if i==0 else [""]*6)
        ax.tick_params(length=0,pad=4)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticks(np.arange(-.5,4,1),minor=True)
        ax.set_yticks(np.arange(-.5,6,1),minor=True)
        ax.grid(which="minor",color="white",linewidth=1)
        ax.tick_params(which="minor",length=0)
        for row in range(6):
            for col in range(4):
                value=values[row,col]
                ax.text(col,row,f"{value:.1f}",ha="center",va="center",size=7.5,color="white" if value>=58 else INK)
        fig.text(.02 if i==0 else .43,.95,title,weight="bold",size=9)
    cb=fig.colorbar(im,cax=fig.add_axes([.25,.075,.31,.025]),orientation="horizontal",ticks=[0,50,100])
    cb.ax.tick_params(length=2,pad=2,labelsize=7)
    cb.outline.set_visible(False)
    fig.text(.14,.078,"TSR (%)",size=8,va="center")
    latency=rows(deploy)
    ax=fig.add_axes([.83,.23,.16,.60])
    fig.text(.74,.95,"(c) Latency (200 requests)",size=9,weight="bold")
    for y,model in zip([2,1,0],["clean_sft","gkd_full","care_opd"]):
        r=next(r for r in latency if r["model"]==model)
        med,p95=float(r["p50_s"]),float(r["p95_s"])
        ax.plot([med,p95],[y,y],color="#B6C2CC",lw=1.5)
        ax.scatter(med,y,color=BLUE,marker="o",s=26,zorder=3)
        ax.scatter(p95,y,color=ORANGE,marker="s",s=26,zorder=3)
    ax.set(yticks=[2,1,0],yticklabels=["Clean-SFT","GKD-Full","CARE-OPD"],xlim=(0,40),ylim=(-.6,2.6),xlabel="Latency (s)")
    ax.set_xticks([0,20,40]);grid(ax)
    ax.scatter([],[],color=BLUE,marker="o",label="Median")
    ax.scatter([],[],color=ORANGE,marker="s",label="P95")
    ax.legend(loc="upper center",bbox_to_anchor=(.4,-.25),ncol=2,handletextpad=.4,columnspacing=.7)
    save(fig,"fig_external_deployment",[source,deploy],{"methods":models,"states":variants,"tracks":tracks,
        "tsr_percent":[a.tolist() for a in matrices],"latency":latency,
        "notes":"Different eligibility rules in (a,b); not a causal preprocessing contrast. Median--P95 is not a confidence interval. Latency excludes the agent loop."})

if __name__=="__main__":
    main()
