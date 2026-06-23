import openml, json
suite = openml.study.get_suite("tabarena-v0.1")
print("n tasks:", len(suite.tasks), flush=True)
rows=[]
for i,tid in enumerate(suite.tasks):
    try:
        t=openml.tasks.get_task(tid, download_data=False, download_qualities=False, download_splits=False)
        did=t.dataset_id
        d=openml.datasets.get_dataset(did, download_data=False, download_qualities=True, download_features_meta_data=False)
        q=d.qualities or {}
        rows.append(dict(task=tid, data=did, name=d.name,
                         rows=int(q.get("NumberOfInstances",-1)),
                         feat=int(q.get("NumberOfFeatures",-1)),
                         cls=int(q.get("NumberOfClasses",-1))))
        print(f"{i+1}/{len(suite.tasks)} {tid} {d.name} rows={rows[-1]['rows']} cls={rows[-1]['cls']}", flush=True)
    except Exception as e:
        print(f"{i+1} ERR {tid}: {str(e)[:50]}", flush=True)
json.dump(rows, open("tabarena_tasks.json","w"), indent=1)
print("SAVED", len(rows), "tasks", flush=True)
