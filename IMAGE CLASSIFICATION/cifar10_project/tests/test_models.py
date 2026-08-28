import torch

from models import BaselineCNN, ResidualBlock, ResNetCNN, build_model


class TestBaselineCNN:
    def test_output_shape(self):
        model = BaselineCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        assert y.shape == (4, 10)

    def test_batch_size_one(self):
        """BatchNorm can fail on batch size 1 in train mode; check eval mode is safe."""
        model = BaselineCNN(num_classes=10).eval()
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10)

    def test_gradients_flow(self):
        model = BaselineCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_dropout_changes_output_in_train_mode(self):
        model = BaselineCNN(num_classes=10, dropout=0.9)
        model.train()
        x = torch.randn(8, 3, 32, 32)
        torch.manual_seed(0)
        y1 = model(x)
        torch.manual_seed(1)
        y2 = model(x)
        assert not torch.allclose(y1, y2)


class TestResidualBlock:
    def test_identity_shortcut_shape(self):
        block = ResidualBlock(32, 32, stride=1)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert y.shape == x.shape

    def test_projection_shortcut_on_channel_change(self):
        block = ResidualBlock(32, 64, stride=1)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert y.shape == (2, 64, 16, 16)

    def test_projection_shortcut_on_downsample(self):
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 16, 16)
        y = block(x)
        assert y.shape == (2, 64, 8, 8)

    def test_residual_actually_adds_signal(self):
        """Sanity check the skip connection is wired up (output != plain conv path)."""
        torch.manual_seed(0)
        block = ResidualBlock(16, 16, stride=1)
        x = torch.randn(1, 16, 8, 8)
        out_with_skip = block(x)

        # Zero out the shortcut's effect by comparing against the conv-only path.
        conv_only = block.relu(block.bn2(block.conv2(block.relu(block.bn1(block.conv1(x))))))
        assert not torch.allclose(out_with_skip, conv_only)


class TestResNetCNN:
    def test_output_shape(self):
        model = ResNetCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        assert y.shape == (4, 10)

    def test_gradients_flow(self):
        model = ResNetCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        loss = model(x).sum()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestBuildModel:
    def test_baseline_factory(self):
        assert isinstance(build_model("baseline"), BaselineCNN)

    def test_resnet_factory(self):
        assert isinstance(build_model("resnet"), ResNetCNN)

    def test_invalid_name_raises(self):
        try:
            build_model("not_a_model")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_num_classes_respected(self):
        model = build_model("baseline", num_classes=100)
        y = model(torch.randn(2, 3, 32, 32))
        assert y.shape == (2, 100)
