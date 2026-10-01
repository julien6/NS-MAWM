"""Format-aware sanitization for binary artifact containers."""
from __future__ import annotations
import io
import json
import re
from pathlib import Path
import torch


def replace_text(text, replacements):
    for old,new in replacements.items():
        text = text.replace(old,new)
    return re.sub(r"/home/[^/\s\"']+", "/home/anonymous", text)


def sanitize_binary(path, replacements):
    suffix = path.suffix.lower()
    buffer = io.BytesIO()
    if suffix in (".pt", ".pth"):
        content = torch.load(path, weights_only=True, map_location="cpu")
        evidence = []
        def clean(value):
            if isinstance(value, str):
                value = replace_text(value,replacements)
                evidence.append(value)
                return value
            if isinstance(value, dict):
                return {clean(k):clean(v) for k,v in value.items()}
            if isinstance(value, list): return [clean(v) for v in value]
            if isinstance(value, tuple): return tuple(clean(v) for v in value)
            if value is None or isinstance(value,(torch.Tensor,int,float,bool)):
                return value
            raise ValueError("Unsupported object in tensor artifact")
        torch.save(clean(content),buffer)
        return buffer.getvalue(), "\n".join(evidence)
    if suffix in (".npy", ".npz"):
        import numpy as np
        loaded = np.load(path,allow_pickle=False)
        arrays = {k:loaded[k] for k in loaded.files} if suffix == ".npz" else {"array":loaded}
        evidence=[]
        for name,array in list(arrays.items()):
            evidence.append(name)
            if array.dtype.kind in "US":
                array=np.array([replace_text(str(v),replacements) for v in array.flat]).reshape(array.shape)
                arrays[name]=array
                evidence.extend(array.flat)
            elif array.dtype.kind == "O":
                raise ValueError("Object arrays are not permitted in anonymous artifacts")
        if suffix == ".npz":
            np.savez_compressed(buffer,**arrays)
            loaded.close()
        else:
            np.save(buffer,arrays["array"],allow_pickle=False)
        return buffer.getvalue(),"\n".join(evidence)
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise ImportError("PDF export requires the optional ns-mawm[export] dependency") from exc
        doc=pymupdf.open(path)
        for page in doc:
            text=page.get_text()
            mapping=dict(replacements)
            for match in re.findall(r"/home/[^/\s]+",text):
                mapping[match]="/home/anonymous"
            for old,new in mapping.items():
                for rectangle in page.search_for(old):
                    page.add_redact_annot(rectangle,text=new,fontsize=8,fill=(1,1,1))
            page.apply_redactions()
        doc.scrub()
        doc.set_metadata({})
        doc.set_toc([])
        evidence="\n".join(page.get_text() for page in doc)
        # Inspect expanded objects too: compressed metadata cannot bypass audit.
        evidence += "\n" + "\n".join(doc.xref_object(i) for i in range(1,doc.xref_length()))
        raw=doc.tobytes(garbage=4,deflate=True,clean=True)
        doc.close()
        return raw,evidence
    if suffix in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        with Image.open(path) as source:
            clean=Image.new(source.mode,source.size)
            clean.putdata(list(source.getdata()))
            clean.save(buffer,format="PNG" if suffix==".png" else "JPEG")
        return buffer.getvalue(),""
    raise ValueError(f"Unsupported binary artifact requires a sanitized export: {path.name}")


def notebook(text):
    obj=json.loads(text)
    obj["metadata"]={}
    for cell in obj.get("cells",[]):
        cell["metadata"]={}
        cell["execution_count"]=None
    return json.dumps(obj,indent=2)
