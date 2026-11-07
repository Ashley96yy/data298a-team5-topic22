"""Verify frozen data and initial SASRec losses before the agreed grid search."""
import argparse
import json
import math
from pathlib import Path
import platform
import torch
from src.models.sasrec import SASRec
from src.data.sequence_dataset import make_dataloaders
from src.training.config import load_config
from src.training.integration_check import load_verified_data,digest
from src.training.trainer import seed_everything
from src.training.train import code_provenance,TIE_POLICY


def check(config_path, output):
    if Path(output).exists():
        raise FileExistsError('Choose a new preflight report path')
    config=load_config(config_path,require_ready=True)
    evaluator,data,hashes=load_verified_data(config['data']['processed_dir'],config)
    training=config['training']
    torch.set_num_threads(2)
    rows=[]
    for hidden in config['search']['hidden_size']:
        seed_everything(42)
        model=SASRec(data['n_items'],max_len=50,hidden_size=hidden,
                     num_heads=training['num_heads'],num_layers=training['num_layers'],dropout=.2).eval()
        loader=make_dataloaders(data['sequences'].iloc[:128],max_len=50,batch_size=128,
                                seed=42,pin_memory=False)['train']
        batch=next(iter(loader))
        with torch.no_grad():
            logits,labels=model.training_logits(batch['input_seq'],batch['target_seq'])
            loss=torch.nn.functional.cross_entropy(logits,labels).item()
        reference=math.log(data['n_items']+1)
        if not math.isfinite(loss) or abs(loss-reference)>.5:
            raise ValueError(f'Unexpected initial loss for hidden size {hidden}: {loss}')
        if torch.count_nonzero(model.item_embedding.weight[0]):
            raise ValueError('Padding row is not zero at initialization')
        rows.append({'hidden_size':hidden,'initial_loss':loss,'uniform_reference':reference})
    report={'status':'PASS','purpose':'initialization_and_data_check_not_final_results',
            'checks':rows,'verified_files':hashes,'tie_policy':TIE_POLICY,'test_evaluated':False,
            'config_sha256':digest(Path(config_path)),**code_provenance()}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with Path(output).open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report['checks'],indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/config.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    check(args.config,args.output)
