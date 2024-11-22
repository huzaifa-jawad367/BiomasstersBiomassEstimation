import torch

def test_pytorch_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Name:", torch.cuda.get_device_name(i))
        # Try a simple tensor operation on GPU
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print("Tensor on GPU:", x)

test_pytorch_cuda()
