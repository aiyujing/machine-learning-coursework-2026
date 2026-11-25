"""Run the same grouped neural experiment on this Mac's MPS GPU."""
from pathlib import Path
import os,ast,certifi
os.environ['SSL_CERT_FILE']=certifi.where()
base=Path(__file__).resolve().parent
os.environ['TORCH_HOME']=str(base/'.torch_home')
s=(base/'train_task3.py').read_text()
s=s.replace("assert torch.cuda.is_available(),'GPU required for this training run'", "from contextlib import nullcontext\nDEVICE='mps'\nassert torch.backends.mps.is_available(),'MPS GPU required'")
s=s.replace("ROOT=next(p.parent for p in Path('/kaggle/input').rglob('fovea_localization_train_GT.csv'))", "ROOT=Path("+repr(str(base/'data'))+")")
s=s.replace("OUT=Path('/kaggle/working')", "OUT=Path("+repr(str(base/'local_artifacts'))+")")
s=s.replace('.cuda()', '.to(DEVICE)').replace("torch.amp.GradScaler('cuda')", "torch.amp.GradScaler('cuda',enabled=False)").replace("torch.autocast('cuda',dtype=torch.float16)", 'nullcontext()').replace('torch.cuda.empty_cache()', 'torch.mps.empty_cache()')
# Preserve the completed coarse search and shorten patience after its plateau.
s=s.replace("HISTORY=[];OOFC", "HISTORY=pd.read_csv(OUT/'task3_training_history.csv').to_dict('records') if (OUT/'task3_training_history.csv').exists() else [];OOFC")
s=s.replace("if epoch>=65 and stale>=35:break", "if epoch>=35 and stale>=25:break")
s=s.replace(" torch.manual_seed(SEED+fold+(100 if stage=='fine' else 0));", " if stage=='coarse' and fold==0 and (OUT/'task3_coarse_fold0.pt').exists():\n  saved=torch.load(OUT/'task3_coarse_fold0.pt',map_location='cpu',weights_only=True);model=Locator().to(DEVICE);model.load_state_dict(saved['state_dict']);model.eval();print('RESUMED_COARSE_BEST',saved['epoch'],saved['valid_rmse'],flush=True);return model,saved['valid_rmse']\n torch.manual_seed(SEED+fold+(100 if stage=='fine' else 0));")
ast.parse(s)
(base/'train_local_generated.py').write_text(s)
exec(compile(s,str(base/'train_local_generated.py'),'exec'))
