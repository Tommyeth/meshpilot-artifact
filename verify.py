"""Offline manuscript checks. No model calls and no physical dispatch."""
from pathlib import Path
import hashlib, json, statistics, subprocess, sys
from meshpilot.artifact_io import read_rows
ROOT=Path(__file__).resolve().parent

def check(condition, message):
    if not condition: raise RuntimeError(message)

def rows(path):
    return read_rows(ROOT/path)

def read(path): return json.loads((ROOT/path).read_text())

def truth(v): return str(v).lower() in ('true','1')

def run(module, *args):
    subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

for line in (ROOT/'checksums.sha256').read_text().splitlines():
    expected, path = line.split('  ', 1)
    check(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected,
          'Data hash mismatch: ' + path)
source='data/decisions.csv.gz'
run('meshpilot.paired_analysis','--requests',source,'--output','reproduced/paired')
run('meshpilot.profile_analysis','--requests',source,'--output','reproduced/profiles')
run('meshpilot.authority_bound','--requests',source,'--output','reproduced/authority-bound')
run('meshpilot.verifier_causal_analysis','--requests',source,'--output','reproduced/verifier','--sample-size','100','--concurrency-trials','100')
p=read('reproduced/paired/summary.json')
check(p['known_id_pairs']==500,'paired count')
check([p[k] for k in ['target_wins','ties','target_losses']]==[98,397,5],'wins/ties/losses')
check(round(100*p['mean_direct_regret'],4)==7.2622,'direct regret')
check(round(100*p['mean_compiled_regret'],4)==0.0867,'compiled regret')
ci=p['compiled_minus_direct_cluster_ci95']
check([round(-100*ci[1],4),round(-100*ci[0],4)]==[2.1084,13.2557],'pooled seed-cluster interval')
profiles=read('reproduced/profiles/summary.json')
check(profiles['known_id_pairs']==500 and profiles['max_recorded_regret_reconstruction_error']==0,'fresh control provenance')
check(round(100*profiles['mean_fresh_balanced_regret'],4)==4.6805,'fresh balanced regret')
comparison=profiles['compiled_vs_balanced']
check([comparison[k] for k in ['target_wins','ties','target_losses']]==[124,374,2],'fresh balanced comparison')
case=profiles['failure_example']
check([case[k] for k in ['seed','scenario','task_id']]==['11','gpu_pressure','task-11-007'],'failure example identity')
check(case['declared_profile']=='resilience' and case['inferred_profile']=='balanced','failure profiles')
check(case['direct_regret']==0 and round(100*case['compiled_regret'],4)==3.2691,'failure example regret')
check(profiles['compiled_losses_vs_direct']['with_profile_error']==5,'loss explanation')
check(profiles['compiled_direct_ties']['same_node']==397 and profiles['compiled_direct_ties']['both_zero_regret']==396,'tie explanation')

a=read('reproduced/authority-bound/summary.json')
check(a['decisions']==600 and a['direct_known_id_exceeding_profile_bound']==36,'authority bound')
v=read('reproduced/verifier/summary.json')
for key,value in {'fault_verifier_rejections':1600,'fault_unverified_invalid_attempts':1600,'fault_fallback_resolutions':1468,'fault_abstention_resolutions':132,'concurrency_trials':100,'naive_precheck_overcommit_exposures':100,'atomic_ledger_overcommits':0}.items():
    check(v[key]==value,key)
physical=rows('data/physical/requests.csv')
check(len(physical)==96 and all(truth(r['success']) for r in physical),'physical completion records')
means={}
for pol,expected in {'round_robin':23.832,'network_greedy':4.531,'joint_fixed':5.168,'vertex_intent_compiled':4.766}.items():
    subset=[r for r in physical if r['policy']==pol]
    check(len(subset)==24,pol+' count')
    means[pol]=statistics.fmean(float(r['actual_end_to_end_ms']) for r in subset)/1000
    check(round(means[pol],3)==expected,pol+' mean')
f=rows('data/physical/failures.csv')
check(len(f)==12 and all(truth(r['final_success']) for r in f),'failure recovery records')
s=rows('data/scaling.csv.gz');timing={}
for n,expected in {12:(.04,.05),50:(.23,.58),100:(.47,.98),200:(1.35,2.79),500:(6.19,10.94),1000:(21.38,30.23)}.items():
    subset=[r for r in s if int(r['nodes'])==n]
    check(len(subset)==600 and all(truth(r['verifier_accepted']) for r in subset),'scale decisions')
    vals=sorted(float(r['planner_ms']) for r in subset)
    timing[n]=[statistics.fmean(vals),float(__import__("numpy").quantile(vals,.99))]
    check(tuple(round(x,2) for x in timing[n])==expected,'recorded timing '+str(n))
report={'status':'PASS','paired_count':500,'wins_ties_losses':[98,397,5],'bound_exceedances':36,'fault_rejections':1600,'atomic_overcommits':0,'physical_recorded_successes':96,'physical_mean_seconds':means,'failure_recorded_successes':12,'recorded_scaling_mean_p99_ms':timing,'scope':'Recomputed frozen-data statistics and deterministic fault/concurrency logic; physical runs and model calls were not repeated.'}
report.update({'artifact':'MeshPilot','mean_reduction_pp':-100*p['mean_compiled_minus_direct'],'pooled_seed_cluster_ci95_reduction_pp':[-100*ci[1],-100*ci[0]],'fresh_balanced_mean_regret_pct':100*profiles['mean_fresh_balanced_regret'],'compiled_vs_balanced_wins_ties_losses':[comparison[k] for k in ['target_wins','ties','target_losses']],'failure_example':{k:case[k] for k in ['seed','scenario','task_id','operator_intent','declared_profile','inferred_profile','direct_regret','compiled_regret']}})
(ROOT/'reproduced').mkdir(exist_ok=True)
(ROOT/'reproduced/results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
