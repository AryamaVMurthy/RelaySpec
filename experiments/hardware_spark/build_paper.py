"""Build a compact paper table/figure from audited, completed GB10 measurements."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


METHODS = ["ar", "native", "source", "relay"]
LABELS = {"ar":"AR", "native":"Native DFlash", "source":"Source reuse", "relay":"RelaySpec"}
COLORS = {"ar":"#7c8187", "native":"#4d8a78", "source":"#d59742", "relay":"#285e8e"}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();root=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True)
    inputs={}
    def read(name):
        path=root/name; data=path.read_bytes();inputs[name]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    done=read("main/complete.json")
    assert done["status"]=="pass" and done["rows"]==256 and done["repeats"]==2 and done["requests"]==32 and not done["profiling"]
    summary=read("main/summary.json");quality=read("main/quality-summary.json")
    hardware=read("hardware-identity.json")
    assert hardware["sys_vendor"]=="MSI" and hardware["product_name"]=="MS-C931"
    assert summary["status"]=="audited" and summary["provenance"]["cap"]==2048
    for m in METHODS:
        for b in ["gsm8k","math500","humaneval"]:
            assert quality["scores"][m][b]["requests"]==8 and quality["scores"][m][b]["unscored"]==0
    memory={}
    for m in ["source","relay"]:
        completion=read(f"memory-{m}/complete.json")
        assert completion["status"]=="pass" and completion["requests"]==4 and completion["source_transformer_resident"]==(m=="source")
        memory[m]=read(f"memory-{m}/summary.json")["overall"][m]
    profile_done=read("timeline-run/complete.json")
    assert profile_done["status"]=="pass" and profile_done["profiling"] and profile_done["rows"]==4
    profile_path=root/"timeline-run/evaluation.jsonl"
    inputs[str(profile_path.relative_to(root))]=hashlib.sha256(profile_path.read_bytes()).hexdigest()
    profile_rows={r["method"]:r for r in map(json.loads,profile_path.read_text().splitlines())}
    csv_path=root/"timeline-summary_nvtx_gpu_proj_sum.csv"
    inputs[csv_path.name]=hashlib.sha256(csv_path.read_bytes()).hexdigest()
    projected={r["Range"].lstrip(":"):float(r["Total Proj Time (ns)"])/1e6 for r in csv.DictReader(csv_path.open())}
    phases={}
    for m in ["source","relay"]:
        names=[m+"::verification_full_target",m+"::draft",m+("::verification_source_trunk" if m=="source" else "::relay")]
        assert all(n in projected for n in names),names
        phases[m]=[projected[n]/profile_rows[m]["output_tokens"] for n in names]
    counters={}
    for m in ["source","relay"]:
        counter_done=read(f"counters-{m}-run/complete.json")
        assert counter_done["status"]=="pass" and counter_done["profiling"] and counter_done["rows"]==1
        path=root/f"counters-{m}-csv.log";raw=path.read_text();inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        start=raw.index('"ID",');table=list(csv.DictReader(io.StringIO(raw[start:])));units=table[0]
        records=[r for r in table if r.get("ID","").isdigit()]
        assert 1<=len(records)<=8
        names=["gpu__time_duration.sum","sm__warps_active.avg.pct_of_peak_sustained_active","sm__throughput.avg.pct_of_peak_sustained_elapsed","lts__throughput.avg.pct_of_peak_sustained_elapsed","dram__bytes.sum.per_second"]
        counters[m]={"sampled_kernels":len(records),"units":{n:units.get(n) for n in names},
                     "rows":[{"kernel":r["Kernel Name"],**{n:r.get(n) for n in names}} for r in records]}
    # A short agreement audit on the fresh isolated runs checks that dropping
    # unused source blocks did not alter the evaluated deployment behavior.
    def raw_rows(name):
        path=root/name;inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return list(map(json.loads,path.read_text().splitlines()))
    main_rows=raw_rows("main/evaluation.jsonl")
    first={(r["method"],r["problem_id"]):r for r in main_rows if r["repeat"]==0}
    scoring_inputs={"records":{r["problem_id"]:r for r in read("main/records.json")},
        "outputs":[{k:r[k] for k in ["method","problem_id","tokens","completion"]} for r in sorted(first.values(),key=lambda r:(r["method"],r["problem_id"]))]}
    assert hashlib.sha256(json.dumps(scoring_inputs,sort_keys=True).encode()).hexdigest()==quality["scoring_inputs_sha256"],"Scored outputs do not match the completed evaluation"
    isolated_agreement={}
    for m in ["source","relay"]:
        rows=raw_rows(f"memory-{m}/evaluation.jsonl")
        isolated_agreement[m]=sum(r["tokens"]==first[m,r["problem_id"]]["tokens"] for r in rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),gridspec_kw={"width_ratios":[1.6,.85,1.05]},layout="constrained")
    tasks=["math500","gsm8k","humaneval","mtbench"];xs=np.arange(4);width=.19
    for i,m in enumerate(METHODS):
        values=[summary["by_workload"][b][m] for b in tasks];positions=xs+(i-1.5)*width
        axes[0].bar(positions,[r["tps"] for r in values],width,label=LABELS[m],color=COLORS[m])
        axes[0].vlines(positions,[r["tps_ci95"][0] for r in values],[r["tps_ci95"][1] for r in values],color="#333333",lw=.8)
    axes[0].set_xticks(xs,["MATH","GSM8K","Code","Dialogue"]);axes[0].set_ylabel("End-to-end tokens/s")
    axes[0].set_title("(a) Matched GB10 throughput",loc="left",fontweight="bold")
    axes[0].legend(fontsize=7.5,ncol=2,frameon=False)
    mem=[memory[m]["peak_allocated_GiB"] for m in ["source","relay"]]
    bars=axes[1].bar([0,1],mem,color=[COLORS["source"],COLORS["relay"]],width=.65)
    axes[1].bar_label(bars,fmt="%.2f",padding=3,fontsize=8.5)
    axes[1].set_xticks([0,1],["Source\nreuse","RelaySpec"]);axes[1].set_ylabel("Peak allocated memory (GiB)")
    axes[1].set_ylim(0,max(mem)*1.18);axes[1].set_title("(b) Isolated deployment",loc="left",fontweight="bold")
    bottom=np.zeros(2)
    for i,(label,color) in enumerate(zip(["Target verification","Drafting","Conditioning"],["#8a939e","#558fa4","#d7a24d"])):
        values=[phases[m][i] for m in ["source","relay"]]
        axes[2].bar([0,1],values,bottom=bottom,label=label,color=color,width=.65);bottom+=values
    axes[2].set_xticks([0,1],["Source\nreuse","RelaySpec"]);axes[2].set_ylabel("GPU-projected ms/token")
    axes[2].set_title("(c) CUDA profile",loc="left",fontweight="bold")
    axes[2].legend(fontsize=7.5,frameon=False)
    for ax in axes:ax.grid(axis="y",alpha=.17);ax.set_axisbelow(True)
    for ext in ["pdf","png"]:
        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None
        fig.savefig(out/f"hardware_spark.{ext}",dpi=200,bbox_inches="tight",metadata=metadata)
    plt.close(fig)
    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",r"Method & Tokens/s & $\times$ AR & J/token & Isolated GiB & Correct M/G/C \\",r"\midrule"]
    for m in METHODS:
        r=summary["overall"][m];q=quality["scores"][m]
        memory_value=f"{memory[m]['peak_allocated_GiB']:.2f}" if m in memory else "--"
        correct="/".join(str(q[b]["correct"]) for b in ["math500","gsm8k","humaneval"])
        energy=r["device_joules_per_token"]
        energy_value=f"{energy:.3f}" if energy is not None else "--"
        lines.append(f"{LABELS[m]} & {r['tps']:.2f} & {r['ratios']['ar']['ratio']:.2f} & {energy_value} & {memory_value} & {correct} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (out/"hardware_spark_table.tex").write_text("\n".join(lines)+"\n")
    metrics_lines=[r"\begin{tabular}{lrrrrrr}",r"\toprule",
        r"Method & GPU util. (\%) & Power (W) & SM (GHz) & Temp. ($^\circ$C) & Progress & AR exact \\",r"\midrule"]
    def metric(r,key,scale=1):
        value=r["gpu_metrics"][key]["mean"]
        return f"{value/scale:.2f}" if value is not None else "--"
    for m in METHODS:
        r=summary["overall"][m]
        values=[metric(r,"utilization.gpu"),metric(r,"power.draw"),metric(r,"clocks.current.sm",1000),metric(r,"temperature.gpu"),f"{r['mean_progress_per_cycle']:.2f}",f"{r['exact_ar']}/32"]
        metrics_lines.append(LABELS[m]+" & "+" & ".join(values)+r" \\")
    metrics_lines += [r"\bottomrule",r"\end{tabular}"]
    (out/"hardware_spark_metrics.tex").write_text("\n".join(metrics_lines)+"\n")
    payload={"input_sha256":inputs,"builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib":matplotlib.__version__,"overall":summary["overall"],"by_workload":summary["by_workload"],
        "memory":memory,"isolated_tokens_match_main":isolated_agreement,"profile_projected_ms_per_token":phases,
        "hardware":{"vendor":hardware["sys_vendor"],"model":hardware["product_name"],"gpu":"NVIDIA GB10","architecture":hardware["architecture"]},
        "sampled_kernel_counters":counters,"quality":quality["scores"],
        "scope":"TPS CI uses paired-request resampling, n=8 per workload. Memory uses 4 fixed requests in isolated processes. Profile uses one warmed MATH request at cap256; projected GPU spans are not unprofiled wall time. Counter replay covers selected projection kernels only. Energy integrates reported device power, not wall power."}
    payload["output_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/"hardware_spark.pdf",out/"hardware_spark.png",out/"hardware_spark_table.tex",out/"hardware_spark_metrics.tex"]}
    (out/"hardware_spark_registry.json").write_text(json.dumps(payload,indent=2)+"\n")


if __name__=="__main__":main()
