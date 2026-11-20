import torch

print(f"torch       {torch.__version__}")
print(f"cuda build  {torch.version.cuda}")
print(f"available   {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device      {props.name} ({props.total_memory / 2**30:.1f} GB)")

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    ok = torch.allclose((a @ b).cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
    print(f"matmul      {'ok' if ok else 'MISMATCH'}")