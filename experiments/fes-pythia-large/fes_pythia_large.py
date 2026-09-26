import argparse,json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

TFS=(0.35,0.50,0.65,0.80,0.95)
SMS=(0.90,1.00,1.10)

def make_windows(tok,text,seq,n,offset=0,extra=17):
    ids=tok(text,return_tensors="pt",add_special_tokens=False)["input_ids"][0]
    need=offset+n*(seq+extra)+seq
    if ids.numel()<need: ids=ids.repeat(math.ceil(need/ids.numel()))
    out=[];p=offset
    for _ in range(n):
        out.append(ids[p:p+seq].clone());p+=seq+extra
    return torch.stack(out)

def data(tok,seq=64,ncal=32,ntest=128,npile=128):
    from datasets import load_dataset
    va=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="validation")
    te=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
    pi=load_dataset("NeelNanda/pile-10k",split="train")
    vat="\n\n".join(x["text"] for x in va if x["text"].strip())
    tet="\n\n".join(x["text"] for x in te if x["text"].strip())
    pit="\n\n".join(x for x in pi[:512]["text"] if x.strip())
    return make_windows(tok,vat,seq,ncal,0),make_windows(tok,tet,seq,ntest,911),make_windows(tok,pit,seq,npile,317)

@torch.inference_mode()
def logits(model,x,batch=16):
    return torch.cat([model(input_ids=x[i:i+batch]).logits[:,:-1].float().cpu() for i in range(0,len(x),batch)],0)

def metrics_from_logits(q,r,x):
    rl=F.log_softmax(r.float(),-1);ql=F.log_softmax(q.float(),-1)
    kl=(rl.exp()*(rl-ql)).sum(-1).mean().item()
    target=x[:,1:].reshape(-1)
    nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),target,reduction="mean").item()
    rnll=F.nll_loss(rl.reshape(-1,rl.shape[-1]),target,reduction="mean").item()
    # window-level NLL difference for uncertainty estimation
    tok_nll=F.nll_loss(ql.reshape(-1,ql.shape[-1]),target,reduction="none").reshape(x.shape[0],-1).mean(-1)
    tok_rnll=F.nll_loss(rl.reshape(-1,rl.shape[-1]),target,reduction="none").reshape(x.shape[0],-1).mean(-1)
    diff=(tok_nll-tok_rnll)
    return {"kl":kl,"nll":nll,"fp_nll":rnll,"ppl":math.exp(min(nll,20)),
            "nll_delta_vs_fp":nll-rnll,"window_delta_mean":diff.mean().item(),
            "window_delta_se":diff.std(unbiased=True).item()/math.sqrt(len(diff))}

def candidates(w):
    w=w.float().cpu();ma=w.abs().mean().item()+1e-12;out=[]
    for tf in TFS:
        s=torch.sign(w)*(w.abs()>=tf*ma)
        den=(s*s).sum().item();base=ma if den==0 else (w*s).sum().item()/den
        for sm in SMS:
            q=(base*sm)*s
            out.append({"q":q.contiguous(),"mse":F.mse_loss(q,w).item(),"tf":tf,"sm":sm})
    return out

def mods(model):return [l.mlp.dense_h_to_4h for l in model.gpt_neox.layers]
def restore(ms,orig):
    for m,w in zip(ms,orig):m.weight.data.copy_(w.to(m.weight.dtype))
def apply(ms,sets,ch):
    for i,m in enumerate(ms):m.weight.data.copy_(sets[i][ch[i]]["q"].to(m.weight.dtype))

@torch.inference_mode()
def eval_model(model,x,ref,batch=16):
    return metrics_from_logits(logits(model,x,batch),ref,x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="results-pythia-large");args=ap.parse_args()
    torch.manual_seed(0);torch.set_num_threads(min(os.cpu_count() or 1,16))
    name="EleutherAI/pythia-14m"
    tok=AutoTokenizer.from_pretrained(name);model=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32);model.eval()
    cal,wiki,pile=data(tok)
    ref_cal=logits(model,cal).half();ref_wiki=logits(model,wiki).half();ref_pile=logits(model,pile).half()
    ms=mods(model);orig=[m.weight.detach().cpu().clone() for m in ms];sets=[candidates(w) for w in orig]
    local=[min(range(15),key=lambda j:sets[i][j]["mse"]) for i in range(len(ms))]

    independent=[]
    for i in range(len(ms)):
        vals=[]
        for j in range(15):
            restore(ms,orig);ms[i].weight.data.copy_(sets[i][j]["q"].to(ms[i].weight.dtype))
            vals.append(eval_model(model,cal,ref_cal)["kl"])
        independent.append(min(range(15),key=lambda j:vals[j]))
        print("independent",i,independent[-1],min(vals),flush=True)
    restore(ms,orig)

    coord=list(independent);history=[]
    for p in range(3):
        changed=False
        for i in range(len(ms)):
            vals=[]
            for j in range(15):
                trial=list(coord);trial[i]=j;apply(ms,sets,trial)
                vals.append(eval_model(model,cal,ref_cal)["kl"])
            b=min(range(15),key=lambda j:vals[j])
            changed|=(b!=coord[i]);coord[i]=b;history.append((p,i,b,vals[b]))
            print("coord",p,i,b,vals[b],flush=True)
        if not changed:break
    restore(ms,orig)

    def ev(ch):
        apply(ms,sets,ch)
        return {"choices":ch,
          "weight_mse":sum(sets[i][ch[i]]["mse"] for i in range(len(ch)))/len(ch),
          "cal":eval_model(model,cal,ref_cal),
          "wiki":eval_model(model,wiki,ref_wiki),
          "pile":eval_model(model,pile,ref_pile)}
    methods={"local":ev(local),"independent":ev(independent),"fes_coordinate":ev(coord)}
    f=methods["fes_coordinate"];ind=methods["independent"];loc=methods["local"]
    summary={
      "cal_tokens":int(cal.numel()),"wiki_tokens":int(wiki.numel()),"pile_tokens":int(pile.numel()),
      "fes_differs_from_independent":coord!=independent,
      "wiki_kl_fes_vs_independent":f["wiki"]["kl"]/ind["wiki"]["kl"],
      "pile_kl_fes_vs_independent":f["pile"]["kl"]/ind["pile"]["kl"],
      "wiki_kl_fes_vs_local":f["wiki"]["kl"]/loc["wiki"]["kl"],
      "pile_kl_fes_vs_local":f["pile"]["kl"]/loc["pile"]["kl"],
      "fes_weight_mse_vs_local":f["weight_mse"]/loc["weight_mse"],
      "wiki_nll_delta_fes":f["wiki"]["nll_delta_vs_fp"],
      "pile_nll_delta_fes":f["pile"]["nll_delta_vs_fp"],
      "wiki_nll_delta_se":f["wiki"]["window_delta_se"],
      "pile_nll_delta_se":f["pile"]["window_delta_se"],
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps({"summary":summary,"methods":methods,"history":history},indent=2))
    print("SUMMARY",json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
