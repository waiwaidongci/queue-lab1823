from .simulation_algorithm import simulate_replicates
from ..utils.fingerprint import fingerprint

def calculate_run(snapshot,customers,warmup,replications,seed,confidence,patience=None):
    s=snapshot['scenario'];p=snapshot['profile']
    result=simulate_replicates(p['arrival_rate'],1/s['mean_service_minutes'],s['servers'],customers,warmup,replications,seed,confidence,patience)
    method='erlangc-fcfs-reneging-t-v2' if patience is not None else 'erlangc-fcfs-independent-t-v1'
    payload={"method":method,"snapshot":snapshot,"customers":customers,"warmup":warmup,"replications":replications,"seed":seed,"confidence":confidence}
    if patience is not None:payload["patience_minutes"]=patience
    result['input_snapshot']=snapshot;result['input_hash']=fingerprint(payload);result['method_version']=method
    return result
