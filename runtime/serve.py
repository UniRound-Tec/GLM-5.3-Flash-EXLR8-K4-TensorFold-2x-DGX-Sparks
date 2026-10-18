"""Container entrypoint; supervise the existing backend and the key-file API adapter."""
import os, pathlib, signal, subprocess, sys, time

def aliases():
    # Preserve the HF snapshot -> ../../blobs resolution by mounting a model repository
    # at its absolute path, then linking the stable loader names to the snapshot.
    for name, variable in (('/model-source', 'MODEL_SOURCE'), ('/draft', 'DRAFT_SOURCE')):
        source = pathlib.Path(os.environ[variable])
        if not source.is_dir(): raise FileNotFoundError(str(source))
        alias = pathlib.Path(name)
        if os.path.lexists(name):
            if not alias.is_symlink() or alias.resolve() != source.resolve():
                raise RuntimeError('Conflicting model alias: '+name)
        else: alias.symlink_to(source, target_is_directory=True)

def command():
    e=os.environ
    cmd=['tensorfold','serve','/exlr8-checkpoint','--tp','2','--rank',e['RANK'],
         '--master',e['MASTER_ADDR'],'--master-port',e.get('MASTER_PORT','29551'),
         '--name',e.get('MODEL_NAME','GLM-5.3-Flash-EXLR8-K4'),
         '--host','127.0.0.1','--port',e.get('BACKEND_PORT','8889'),
         '--drafter',e['DRAFT_SOURCE'],'--context',e.get('CONTEXT','1048576'),
         '--parallel',e.get('PARALLEL','4'),'--max-tokens',e.get('MAX_TOKENS','32768'),
         '--thinking']
    if e['RANK']=='0' and e.get('VISION','1')=='1':cmd.append('--vision')
    return cmd

def main():
    aliases()
    children=[]
    def stop(*_):
        for p in children:
            if p.poll() is None:p.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        children.append(subprocess.Popen(command()))
        if os.environ['RANK']=='0':
            children.append(subprocess.Popen([sys.executable,'/opt/recipe/runtime/auth_proxy.py']))
        while all(p.poll() is None for p in children):time.sleep(.5)
        return next((p.returncode for p in children if p.returncode is not None),1)
    finally:
        stop()
        for p in children:
            try:p.wait(timeout=25)
            except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':sys.exit(main())
