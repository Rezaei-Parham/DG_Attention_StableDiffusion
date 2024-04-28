import torch


def compute_loss(attention_maps,
                 indices_to_alter,
                 i) -> torch.Tensor:
    attention_map_1 = attention_maps[:, :, indices_to_alter[0]]
    attention_map_2 = attention_maps[:, :, indices_to_alter[1]]
    loss = torch.sum(attention_map_1) + torch.sum(attention_map_2)
    return loss