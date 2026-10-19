"""CPU-only packaging checks: no SSH, GPU, requests or deployment."""
import importlib.util,json,os,pathlib,sys,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=module('recipe','scripts/recipe.py')
s=module('serve','runtime/serve.py')
a=module('auth_proxy','runtime/auth_proxy.py')

class RecipeTests(unittest.TestCase):
    def test_rank_mounts_and_secrets(self):
        c=json.loads((ROOT/'config/example.json').read_text())
        p={'data_dir':'/recipe-data','model_dir':'/models/source','draft_dir':'/drafts/snapshots/'+r.DRAFTS[-1]}
        head=r.command(c,0,p,['/models']);worker=r.command(c,1,p,['/models'])
        self.assertTrue(any('dst=/run/secrets/api-key' in x for x in head))
        self.assertFalse(any('api-key' in x for x in worker))
        self.assertIn('RANK=1',worker)
        self.assertIn('DRAFT_SOURCE='+p['draft_dir'],head)
        self.assertIn('--restart=no',head)
        self.assertTrue(any('src=/drafts,dst=/drafts,readonly' in x for x in head))
    def test_vision_head_only_and_actual_draft(self):
        env={'RANK':'0','MASTER_ADDR':'10.0.0.1','DRAFT_SOURCE':'/drafts/snapshots/'+r.DRAFTS[-1],'VISION':'1'}
        with patch.dict(os.environ,env,clear=True):
            cmd=s.command();self.assertIn('--vision',cmd)
            self.assertEqual(cmd[cmd.index('--drafter')+1],env['DRAFT_SOURCE'])
        env['RANK']='1'
        with patch.dict(os.environ,env,clear=True):self.assertNotIn('--vision',s.command())
    def test_explicit_benchmark_seed_preserved(self):
        raw=b'{"seed":42,"messages":[]}'
        self.assertEqual(a.randomize_seed(raw,'POST','/v1/chat/completions'),raw)
        self.assertIsInstance(json.loads(a.randomize_seed(b'{}','POST','/v1/chat/completions'))['seed'],int)
    def test_benchmark_evidence_math(self):
        e=json.loads((ROOT/'evidence/benchmarks.json').read_text())
        for row in e['decode']:
            self.assertEqual(row['usage']['completion_tokens'],512)
            self.assertTrue(row['stats']['tf_trials']['verify8'])
            self.assertEqual(row['stats']['tf_trials']['draft_checkpoint'],31)
        code=[x for x in e['decode'] if x['name']=='mia_code']
        best=max((x['usage']['completion_tokens']-1)/x['stats']['decode_s'] for x in code)
        self.assertAlmostEqual(best,119.0587138863,places=8)
    def test_missing_retired_draft_patch_is_included(self):
        source=(ROOT/'vendor/tensorfold/families/glm5_next/cuda/draft_checkpoints.py').read_text()
        self.assertIn("if not (path/'model.safetensors').is_file():",source)
        self.assertIn('Retired historical checkpoint',source)
    @patch.dict(os.environ, {'TF_GLM_TRIAL_BANKS':'1'})
    def test_prefill_defaults_protocol_and_cache_separation(self):
        control=module('packaged_trial_control','vendor/tensorfold/families/glm5_next/cuda/trial_control.py')
        defaults=json.loads((ROOT/'config/defaults.json').read_text())
        parsed=control.parse(defaults)
        self.assertEqual(parsed['bf16_prefill'],3)
        self.assertEqual(parsed['prefill_pieces'],1)
        self.assertEqual(parsed['prefill_schedule'],2)
        self.assertFalse(parsed['prefill_atomic'])
        self.assertFalse(parsed['prefill_profile'])
        self.assertFalse(parsed['mx_prefill'])
        self.assertEqual(control.unpack(control.pack(parsed)),parsed)
        original=control.parse({**defaults,'bf16_prefill':0,'prefill_pieces':0,'prefill_schedule':0})
        control.ACTIVE=parsed
        retained_tag=control.tag()
        control.ACTIVE=original
        self.assertNotEqual(retained_tag,control.tag())
        self.assertEqual(control.unpack(control.pack(parsed)[:25])['prefill_schedule'],0)
    def test_dated_prefill_evidence(self):
        e=json.loads((ROOT/'evidence/prefill-20261019.json').read_text())
        for name in ('bf16_paired','schedule_paired','retained','rejected_atomic'):
            for row in e[name]:
                self.assertEqual(row['measurement_date'],'2026-10-19')
                self.assertTrue(row['isolated'])
                self.assertEqual(row['stats']['cached'],0)
                self.assertEqual(row['usage']['completion_tokens'],32)
                self.assertAlmostEqual(row['prefill_tps'],row['usage']['prompt_tokens']/row['stats']['prefill_s'])
        retained={r['nominal']:r for r in e['retained']}
        self.assertAlmostEqual(retained[32768]['prefill_tps'],1894.7058585881905)
        self.assertAlmostEqual(retained[196608]['prefill_tps'],1854.565995175524)
        for row in retained.values():
            self.assertFalse(row['stats']['tf_trials']['prefill_atomic'])
    def test_prefill_source_snapshot(self):
        import hashlib
        e=json.loads((ROOT/'evidence/prefill-source-20261019.json').read_text())
        base=ROOT/'vendor/tensorfold/families/glm5_next/cuda'
        for name,digest in e['runtime_files'].items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),digest,name)

if __name__=='__main__':unittest.main()
