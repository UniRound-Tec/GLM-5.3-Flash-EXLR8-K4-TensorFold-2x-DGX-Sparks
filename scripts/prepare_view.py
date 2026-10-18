"""Build a metadata-only EXLR8 K4 view; original weights are never modified."""
import argparse,json,math,struct
from pathlib import Path
REV='3117e3b85744ca91c49fcbe95b46c33f31091829'
DT={'F16':2,'BF16':2,'F32':4,'I16':2,'I8':1,'I64':8,'U8':1,'I32':4,'U64':8}
def header(p):
    with Path(p).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>64*1024*1024:raise ValueError('Unexpected tensor header size')
        return json.loads(f.read(n)),n+8
def mount_root(p):
    p=Path(p).resolve()
    for a in (p,*p.parents):
        if a.name.startswith('models--') and a.parent.name=='hub':return str(a)
    return str(p if p.is_dir() else p.parent)
def create(source,output,vision=None):
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().absolute()
    if source==Path('/'):raise ValueError('Model source must be a dedicated directory')
    cfg=json.loads((source/'config.json').read_text());text=cfg.get('text_config',cfg)
    expected={'hidden_size':4096,'num_hidden_layers':45,'n_routed_experts':288,'moe_intermediate_size':2048}
    if any(text.get(k)!=v for k,v in expected.items()):raise ValueError('Expected GLM Flash geometry')
    if source.parent.name=='snapshots' and source.name!=REV:raise ValueError('Wrong pinned target revision')
    h={};roots={mount_root(source)}
    def add(k,shape,dtype):h[k]={'shape':shape,'dtype':dtype,'data_offsets':[0,math.prod(shape)*DT[dtype]],'split':False}
    dense=list(sorted((source/'dense').glob('*.safetensors')))
    if len(dense)!=47:raise ValueError('Expected 47 dense files, including MTP metadata and vocab')
    for p in dense:
        hh,base=header(p);vocab='vocab' in p.name
        pre='' if vocab else f'model.language_model.layers.{int(p.stem.split("-")[-1])}.'
        with p.open('rb') as f:
            for key,info in hh.items():
                if key.startswith('__'):continue
                if key.endswith('.record'):
                    f.seek(base+info['data_offsets'][0]);fmt,n,k,*_=struct.unpack('<8q',f.read(64))
                    name=pre+key[:-7]+'.weight'
                    if vocab and key.startswith('embed_tokens'):name='model.language_model.'+name
                    add(name,[n,k],'F32' if fmt==7 else 'BF16')
                elif not any(key.endswith('.'+s) for s in ('checksum','trellis','suh','svh','fp8','e8m0','scale','bf16','fp32','q4','s4')):
                    add(pre+key if not vocab else 'model.language_model.'+key,info['shape'],info['dtype'])
    for layer in range(3,46):
        ix=source/f'experts/exlr8-layer-{layer:03d}.index.safetensors'
        packed=source/f'experts/exlr8-layer-{layer:03d}.k4.safetensors'
        if not packed.is_file() or header(ix)[0].get('k4.index',{}).get('shape')!=[288,3,16,2]:raise ValueError(f'Missing/bad K4 layer {layer}')
        for e in range(288):
            for proj,k,n in (('gate',4096,2048),('up',4096,2048),('down',2048,4096)):
                pre=f'model.language_model.layers.{layer}.mlp.experts.{e}.{proj}_proj.'
                add(pre+'trellis',[k//16,n//16,64],'I16');add(pre+'suh',[k],'F16');add(pre+'svh',[n],'F16')
        for proj,k,n in (('gate',4096,2048),('up',4096,2048),('down',2048,4096)):
            add(f'model.language_model.layers.{layer}.mlp.shared_experts.{proj}_proj.weight',[n,k],'BF16')
    links={}
    for name in ('tokenizer.json','tokenizer_config.json','generation_config.json','chat_template.jinja'):
        if (source/name).is_file():links[name]=source/name
    mapping={};vision_bytes=0;processors={}
    if vision:
        vision=Path(vision).expanduser().resolve();vcfg=json.loads((vision/'config.json').read_text())
        meta={'label2id','return_dict','chunk_size_feed_forward','_name_or_path','id2label','output_attentions','is_encoder_decoder','dtype','output_hidden_states'}
        va,vb=cfg['vision_config'],vcfg['vision_config']
        if any(va.get(k)!=vb.get(k) for k in (set(va)|set(vb))-meta):raise ValueError('Vision geometry mismatch')
        if any(cfg[k]!=vcfg[k] for k in ('image_token_id','video_token_id')):raise ValueError('Vision token mismatch')
        idx=json.loads((vision/'model.safetensors.index.json').read_text())['weight_map']
        selected={k:v for k,v in idx.items() if k.startswith('model.visual.')}
        if len(selected)!=347:raise ValueError('Expected 347 vision tensors')
        for i,name in enumerate(sorted(set(selected.values()))):
            p=(vision/name).resolve();hh,_=header(p);alias=f'vision-source-{i:03d}.safetensors'
            links[alias]=p;roots.add(mount_root(p))
            for k,v in selected.items():
                if v!=name:continue
                info=hh[k]
                if info['dtype']!='BF16':raise ValueError('Expected source BF16 vision')
                h[k]={**info,'split':False};mapping[k]=alias;vision_bytes+=info['data_offsets'][1]-info['data_offsets'][0]
        if vision_bytes!=1127254016:raise ValueError('Unexpected vision bytes')
        for name in ('processor_config.json','preprocessor_config.json','video_preprocessor_config.json'):
            if (vision/name).is_file():processors[name]=(vision/name).read_bytes()
    cfg['quantization_config']={'quant_method':'exl3','bits':4,'group_size':128,'source_format':'exlr8-e4m3-v2','source_revision':REV}
    cfg.pop('quantization',None)
    signature={'source':str(source),'vision':str(vision or ''),'revision':REV}
    marker=output/'recipe-view.json'
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text())!=signature:raise ValueError('Existing view belongs to another configuration; choose a new data directory')
    else:output.mkdir(parents=True)
    for name,p in links.items():
        dest=output/name
        if dest.is_symlink():
            if dest.resolve()!=p.resolve():raise ValueError('View link changed: '+name)
        elif dest.exists():raise ValueError('Unexpected file in metadata view: '+name)
        else:dest.symlink_to(p)
    for name,data in processors.items():(output/name).write_bytes(data)
    for name,data in {'config.json':cfg,'exlr8-source.json':{'path':str(source),'revision':REV},'exlr8-runtime-headers.json':h,'model.safetensors.index.json':{'metadata':{'total_size':vision_bytes},'weight_map':mapping},'recipe-view.json':signature}.items():
        (output/name).write_text(json.dumps(data),encoding='utf-8')
    report={'logical_tensors':len(h),'vision_tensors':len(mapping),'vision_bytes':vision_bytes,'mount_roots':sorted(roots)}
    (output/'recipe-mounts.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--vision')
    a=p.parse_args();print(json.dumps(create(a.source,a.output,a.vision)))
