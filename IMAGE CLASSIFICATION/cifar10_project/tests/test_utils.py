import os
import tempfile

import torch

from models import BaselineCNN
from utils import EarlyStopper, load_checkpoint, save_checkpoint, set_seed


class TestSetSeed:
    def test_reproducible_random_tensor(self):
        set_seed(123)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_reproducible_model_init(self):
        set_seed(7)
        m1 = BaselineCNN()
        set_seed(7)
        m2 = BaselineCNN()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)


class TestEarlyStopper:
    def test_stops_after_patience_exceeded(self):
        stopper = EarlyStopper(patience=3)
        accs = [0.5, 0.6, 0.6, 0.6, 0.6] 
        stopped_at = None
        for i, a in enumerate(accs):
            if stopper.step(a):
                stopped_at = i
                break
        assert stopped_at == 4  

    def test_no_stop_when_improving(self):
        stopper = EarlyStopper(patience=2)
        accs = [0.1, 0.2, 0.3, 0.4, 0.5]
        for a in accs:
            assert not stopper.step(a)


class TestCheckpointing:
    def test_save_and_load_roundtrip(self):
        model = BaselineCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(path, model, optimizer, None, epoch=5,
                             best_val_acc=0.42, history={"val_acc": [0.42]})
            assert os.path.exists(path)

            new_model = BaselineCNN()
            ckpt = load_checkpoint(path, new_model, map_location="cpu")
            assert ckpt["epoch"] == 5
            assert ckpt["best_val_acc"] == 0.42
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.equal(p1, p2)
