import torch

def position_interpolation(pos_ids, max_train_len, max_new_len):
    scale = max_train_len / max_new_len
    return pos_ids * scale

if __name__ == "__main__":
    pos_ids = torch.arange(0, 16)
    scaled = position_interpolation(pos_ids, max_train_len=2048, max_new_len=4096)
    print(f">>> pos_ids = torch.arange(0, 16)")
    print(f">>> scaled = position_interpolation(pos_ids, max_train_len=2048, max_new_len=4096)")
    print(f">>> scaled\n{scaled}")
    print(f"# Scale factor = 2048 / 4096 = 0.5, positions compressed into training range")