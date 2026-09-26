import argparse, importlib.util, json, math, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_base():
    p=Path("experiments/fes-transformer/fes_transformer.py")
    s=importlib.util.spec_from_file_location("fes_base",p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def make_windows(tok, seq_len, n_cal, n_test, offset):
    from datasets import load_dataset
    ds=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    text="\n\n".join(x["text"] for x in ds if x["text"].strip())
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+(n_cal+n_test)*(seq_len+7)+seq_len
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    windows=[]; pos=offset
    for _ in range(n_cal+n_test):
        windows.append(ids[pos:pos+seq_len].clone()); pos+=seq_len+7
    return windows[:n_cal],windows[n_cal:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--passes",type=int,default=2)
    ap.add_argument("--out",default="results-split")
    args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(min(os.cpu_count() or 1,16))
    base=load_base()
    model_name="EleutherAI/pythia-14m"
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float32); model.eval()
    modules=base.get_target_modules(model)
    originals=[m.weight.detach().float().cpu().clone() for _,m in modules]
    candidate_sets=[base.ternary_candidates(w) for w in originals]
    cal_w,test_w=make_windows(tok,64,4,8,args.offset)
    fp_cal,fp_test,_=base.cache_fp(model,cal_w,test_w,32)

    local=[min(range(len(cs)),key=lambda j:cs[j]["weight_mse"]) for cs in candidate_sets]

    independent=[]
    for i,cs in enumerate(candidate_sets):
        best_j=None; best_kl=None
        for j,c in enumerate(cs):
            base.restore(modules,originals)
            modules[i][1].weight.data.copy_(c["q"].to(modules[i][1].weight.dtype))
            kl=base.metrics(model,cal_w,fp_cal)["kl_to_fp"]
            if best_kl is None or kl<best_kl: best_kl,best_j=kl,j
        independent.append(best_j)
        print(f"independent layer={i} choice={best_j} cal_kl={best_kl:.8g}",flush=True)
    base.restore(modules,originals)

    choices=list(independent); history=[]
    for p in range(args.passes):
        changed=False
        for i,cs in enumerate(candidate_sets):
            best_j=choices[i]; best_kl=None
            for j in range(len(cs)):
                trial=list(choices); trial[i]=j
                base.apply_choices(modules,candidate_sets,trial)
                kl=base.metrics(model,cal_w,fp_cal)["kl_to_fp"]
                if best_kl is None or kl<best_kl: best_kl,best_j=kl,j
            if best_j!=choices[i]: changed=True
            choices[i]=best_j
            history.append({"pass":p,"layer":i,"choice":best_j,"cal_kl":best_kl})
            print(f"coordinate pass={p} layer={i} choice={best_j} cal_kl={best_kl:.8g}",flush=True)
        if not changed: break
    base.restore(modules,originals)

    methods={
      "local":base.evaluate(model,modules,originals,candidate_sets,local,cal_w,test_w,fp_cal,fp_test),
      "independent":base.evaluate(model,modules,originals,candidate_sets,independent,cal_w,test_w,fp_cal,fp_test),
      "coordinate":base.evaluate(model,modules,originals,candidate_sets,choices,cal_w,test_w,fp_cal,fp_test),
    }
    s={
      "offset":args.offset,
      "local_choices":local,
      "independent_choices":independent,
      "coordinate_choices":choices,
      "coordinate_vs_independent_test_kl_ratio":methods["coordinate"]["test_kl_to_fp"]/methods["independent"]["test_kl_to_fp"],
      "coordinate_vs_local_test_kl_ratio":methods["coordinate"]["test_kl_to_fp"]/methods["local"]["test_kl_to_fp"],
      "coordinate_vs_independent_weight_mse_ratio":methods["coordinate"]["weight_mse"]/methods["independent"]["weight_mse"],
      "heldout_gain_vs_independent":methods["coordinate"]["test_kl_to_fp"]<methods["independent"]["test_kl_to_fp"],
      "core_condition":methods["coordinate"]["test_kl_to_fp"]<methods["local"]["test_kl_to_fp"] and methods["coordinate"]["weight_mse"]>methods["local"]["weight_mse"],
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":s,"methods":methods,"history":history},indent=2))
    print("SUMMARY",json.dumps(s,sort_keys=True),flush=True)

if __name__=="__main__": main()
