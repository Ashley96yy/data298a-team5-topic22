"""Selected artifacts remain loadable after relocation; corrupt/mismatched artifacts fail."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from src.models.sasrec import SASRec
from src.training.checkpoints import load_selected_model
from src.training.config import expand_grid, load_config
from src.training.integration_check import digest
from src.training.train import TIE_POLICY
from src.training.trainer import TrainingSettings, fit, seed_everything
from src.data.sequence_dataset import make_dataloaders
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class SelectedCheckpointTests(unittest.TestCase):
    def test_restored_predictions_and_relocated_artifacts(self):
        torch.set_num_threads(2)
        seed_everything(42)
        config = load_config(ROOT / 'configs/config.json', require_ready=True)
        config['training'].update(num_heads=4, seeds=[42], optimizer='AdamW')
        selected = expand_grid(config)[0]
        model = SASRec(10, max_len=50, hidden_size=64, num_heads=4, num_layers=2, dropout=.2)
        data = pd.DataFrame({'user_idx':[0,1], 'train_items':[[1,2,3],[2,3,4]],
                             'val_item':[5,6], 'test_item':[7,8]})
        loader = make_dataloaders(data,max_len=50,batch_size=2,seed=42,pin_memory=False)['train']
        metadata = {'model_hyperparameters': selected['model_hyperparameters'],
                    'model_layers':2,'model_heads':4,'max_seq_len':50,
                    'selection_split':'val','selection_metric':'NDCG@10','sampled':False,
                    'tie_policy':TIE_POLICY,'manifest_sha256':'synthetic-fixture'}
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'original'
            fit(model,loader,lambda _: {'NDCG@10':.2},
                TrainingSettings(epochs=1,patience=2,optimizer='AdamW',learning_rate=.001,
                                 weight_decay=.00001,seed=42),run,metadata=metadata)
            moved=Path(temp)/'moved';moved.mkdir()
            shutil.copyfile(run/'best.pt',moved/'best_checkpoint.pt')
            (moved/'best_config.json').write_text(json.dumps(selected))
            selection={'status':'PASS','completed_runs':8,'selection_split':'val',
                       'selection_metric':'NDCG@10','sampled':False,'test_evaluated':False,
                       'seed':42,'tie_policy':TIE_POLICY,'best_epoch':1,
                       'best_validation_NDCG@10':.2,
                       'checkpoint_sha256':digest(moved/'best_checkpoint.pt')}
            (moved/'selection.json').write_text(json.dumps(selection))
            restored,cp,_=load_selected_model(moved)
            tokens=next(iter(loader))['input_seq']
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model.next_item_scores(tokens),restored.next_item_scores(tokens),rtol=0,atol=0)
            self.assertEqual(cp['metadata']['manifest_sha256'],'synthetic-fixture')
            selected['model_hyperparameters']['dropout']=.5
            (moved/'best_config.json').write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError,'metadata'):
                load_selected_model(moved)
            with (moved/'best_checkpoint.pt').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_selected_model(moved)
