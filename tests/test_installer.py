"""Offline tests for install integrity and configuration; no experimental data."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'app'))
import install
import runtime_config as rc

class InstallationTests(unittest.TestCase):
    def test_bundled_integrity(self):
        install.verify_bundled()

    def test_config_has_no_machine_specific_paths(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'TWOPHOTON_STATE_DIR':folder}):
            config=rc.runtime_defaults()
            self.assertEqual(Path(config['cellpose_model']),Path(folder)/'models/cellpose/cpsam')
            source=(ROOT/'app/runtime_config.py').read_text(encoding='utf-8')
            self.assertNotIn('Users\\Admin',source)
            self.assertNotIn('Anaconda3_py3.9',source)
            self.assertEqual(config['device'],'cpu')

    def test_session_rebinding_keeps_analysis_parameters(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'TWOPHOTON_STATE_DIR':folder}):
            model=Path(folder)/'custom.pt'; model.write_bytes(b'test')
            (Path(folder)/'runtime.json').write_text(json.dumps({'device':'cpu','modern_python':str(model)}))
            got=rc.rebind_runtime(dict(fs=30.,tau=.45,profile_checkpoint=str(model),
                                      profile_root=str(Path(folder)/'missing-profile'),profile_python=str(Path(folder)/'missing-python.exe'),
                                      suite2p_backend='gpu',device='cuda'))
            self.assertEqual(got['tau'],.45); self.assertEqual(got['fs'],30.)
            self.assertEqual(got['profile_checkpoint'],str(model))
            self.assertTrue(Path(got['profile_root']).is_dir())
            self.assertEqual(got['device'],'cpu')

    def test_unowned_environment_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'existing'; target.mkdir()
            with self.assertRaisesRegex(RuntimeError,'unmanaged'):
                install.make_env('unused',target,'3.8',{})

    def test_wrong_model_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'wrong'; source.write_bytes(b'not a checkpoint')
            target=Path(folder)/'models/cpsam'
            with self.assertRaises(ValueError):
                install.download_model(target,source)
            self.assertFalse(target.exists())

    def test_repository_does_not_ship_data_or_personal_config(self):
        for name in ('preferences.ini','session.npy','Fall.mat','runtime.json'):
            self.assertFalse((ROOT/'app'/name).exists())

if __name__=='__main__': unittest.main()

