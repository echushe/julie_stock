import torch

print(f'There are {torch.cuda.device_count()} GPUs.\n')  # Number of GPUs
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda:{}'.format(i))
    print(f'using device: {device}')
    print()

