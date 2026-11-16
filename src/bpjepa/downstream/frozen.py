"""Frozen-encoder spotting: cached per-pair spatial features plugged into the native ViViT spotting model."""
from bpjepa.downstream import compat  # noqa: F401
import hashlib,json,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from eztorch.datamodules.soccernet import SoccerNetDataModule
from eztorch.datasets.decoders.frame_soccernet_video import FrameSoccerNetVideo
from eztorch.models.trunks.transformer_token_handler import create_vitransformer_token_handler_model

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def spatial_keys(state):
    return [k for k in state if k in {"spatial_cls_token","pos_embed"} or k.startswith(("patch_embed.","spatial_transformer."))]

def load_pretrained(transformer,source,expected_step=81900):
    """Online (context) encoder of the fixed final pretraining checkpoint into the spatial subset; nothing else is loaded."""
    if source.get("sha256") not in (None,sha(source["path"])): raise RuntimeError("pretrained checkpoint hash mismatch")
    d=torch.load(source["path"],map_location="cpu",weights_only=False)
    if expected_step is not None and d["global_step"]!=expected_step: raise RuntimeError(f"expected the final pretraining checkpoint (step {expected_step}), got step {d['global_step']}")
    from bpjepa.downstream.initialization import source_key_for_target
    current=transformer.state_dict()
    context={k.removeprefix("context_encoder."):v for k,v in d["model"].items() if k.startswith("context_encoder.")}
    mapped={k:context[source_key_for_target(k).removeprefix("trunk.")] for k in spatial_keys(current)}
    if len(mapped)!=len(context): raise RuntimeError("incomplete spatial mapping")
    transformer.load_state_dict({**current,**mapped},strict=True)
    assert all(torch.equal(transformer.state_dict()[k],v) for k,v in mapped.items())
    for name,p in transformer.named_parameters():
        if name in mapped: p.requires_grad_(False)
    transformer.freeze_spatial=True
    transformer.freeze_spatial_transformer()
    transformer.pos_drop.eval()
    return {"checkpoint":source,"spatial_tensors":len(mapped),"exact":True}

def cached_spatial(self,x):
    if x.ndim!=3 or x.shape[1:]!=(32,192): raise ValueError("native cache input must be Bx32x192")
    if not torch.isfinite(x).all(): raise ValueError("nonfinite cached features")
    return x.detach(),x.shape[0]

def cached_train(self,mode=True):
    type(self).train(self,mode)
    self.pos_drop.eval()
    return self

def create_cached_trunk(**kwargs):
    trunk=create_vitransformer_token_handler_model(**kwargs)
    trunk.transformer.forward_spatial=types.MethodType(cached_spatial,trunk.transformer)
    trunk.transformer.train=types.MethodType(cached_train,trunk.transformer)
    return trunk

def frozen_head_hashes(model):
    from bpjepa.downstream.initialization import state_hashes,tensor_sha256
    t=model.trunk.transformer
    return {"temporal":state_hashes(t.temporal_transformer),"time_embed":tensor_sha256(t.time_embed),"classifier":state_hashes(model.head_class)}

class PairCache:
    def __init__(self,registry,expected_encoder=None):
        self.path=Path(registry)
        self.doc=json.loads(self.path.read_text())
        if self.doc.get("artifact_kind")!="frozen_spatial_full_half_cache": raise ValueError("full-half cache registry required")
        if self.doc["dtype"]!="float32" or self.doc["width"]!=192: raise ValueError("unsupported cache contract")
        if expected_encoder and self.doc["encoder"].get("sha256") not in (None,expected_encoder.get("sha256")): raise RuntimeError("cache encoder mismatch")
        self.arrays={}
    def bank(self,half):
        half=str(Path(half))
        if half not in self.doc["halves"]: raise KeyError(f"half not in the cache (use the same --frames-dir as extraction): {half}")
        if half not in self.arrays:
            entry=self.doc["halves"][half]
            if entry.get("filled_rows")!=entry["frames"]+1: raise ValueError("incomplete half cache")
            path=self.path.parent/entry["file"]
            if sha(path)!=entry["sha256"]: raise RuntimeError("cache half hash mismatch")
            a=np.load(path,mmap_mode="r")
            if a.dtype!=np.float32 or a.shape!=(entry["frames"]+1,192): raise ValueError("invalid cache bank")
            self.arrays[half]=a
        return self.arrays[half]
    def gather(self,half,indices):
        bank=self.bank(half)
        n=bank.shape[0]-1
        ids=torch.as_tensor(indices,dtype=torch.long).reshape(-1,2)
        a,b=ids[:,0],ids[:,1]
        if bool(((a<0)|(b>=n)).any()): raise ValueError("pair outside half")
        adjacent=b==a+1
        first=(a==0)&(b==0)
        final=(a==n-1)&(b==n-1)
        if not bool((adjacent|first|final).all()): raise ValueError("missing frame pair")
        rows=torch.where(first,n-1,torch.where(final,n,a))
        return torch.from_numpy(np.array(bank[rows.numpy()],copy=True))

class CachedVideo:
    def __init__(self,cache,**kwargs):
        self.native=FrameSoccerNetVideo(**kwargs)
        self.half=kwargs["half_path"]
        self.cache=cache
    def close(self):
        self.native.close()
    def get_clip(self,start,end):
        ts,ids,source_ids=self.native.get_timestamps_and_frame_indices(start,end)
        if len(ts)!=64 or len(ts)%2: raise ValueError("requires native64-frame clips")
        return {"video":self.cache.gather(self.half,source_ids),"clip_start":ts[0].item(),
                "clip_end":ts[-1].item(),"clip_duration":(ts[-1]-ts[0]).item(),
                "frame_indices":ids,"fps_video_frame_indices":source_ids,"timestamps":ts}

class CachedHandler:
    def __init__(self,registry): self.cache=PairCache(registry)
    def video_from_path(self,decoder,**kwargs):
        if str(decoder) not in {"frame","DecoderType.FRAME"}: raise ValueError("only native frame cache")
        return CachedVideo(self.cache,**kwargs)

class CachedDataModule(SoccerNetDataModule):
    def __init__(self,cache_registry,**kwargs):
        if kwargs.get("test") is not None: raise ValueError("test forbidden")
        self.cache_registry=cache_registry
        super().__init__(**kwargs)
    def setup(self,stage=None):
        if stage not in {"fit","validate"}: raise ValueError("test/other stages forbidden")
        super().setup("fit")
        for name in ("train_dataset","val_dataset"):
            ds=getattr(self,name,None)
            if ds is not None: ds._video_path_handler=CachedHandler(self.cache_registry)

def config_for_cache(config,registry,output):
    from copy import deepcopy
    c=deepcopy(config)
    c["datamodule"]["_target_"]="bpjepa.downstream.frozen.CachedDataModule"
    c["datamodule"]["cache_registry"]=str(registry)
    c["datamodule"]["test"]=None
    c["model"]["_target_"]="eztorch.models.soccernet_spotting.SoccerNetSpottingModel"
    c["model"].pop("spatial_init",None)
    c["model"]["trunk"]["_target_"]="bpjepa.downstream.frozen.create_cached_trunk"
    c["model"]["trunk"]["transformer"]["freeze_spatial"]=True
    reduction={"_target_":"eztorch.transforms.video.soccernet.BatchReduceTimestamps"}
    c["model"]["train_transform"]=reduction
    c["model"]["val_transform"]=reduction
    c["model"]["test_transform"]=None
    c["model"]["save_val_preds_path"]=str(Path(output)/"val_preds")
    c["model"]["save_test_preds_path"]=str(Path(output)/"test_forbidden")
    for key in ("model_checkpoint","epoch_checkpoint"):
        if key in c["callbacks"]:c["callbacks"][key]["dirpath"]=str(Path(output)/"checkpoints")
    return c
