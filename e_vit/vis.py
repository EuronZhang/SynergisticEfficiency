import torch


def idx_mapping(selected_idx_list, num_tokens=196):
    """
    selected_idx_list: a list of tensors, tensor: [B, L], B-batch size, L-num of selected token
    return: 
    A list of slected_idx mapping back to image idx
    """
    B, _ = selected_idx_list[0].shape
    idx = torch.arange(num_tokens).expand(B, -1)
    mapping_idx = []

    for selected_idx in selected_idx_list:
        idx = torch.gather(idx, dim=1, index=selected_idx)
        mapping_idx.append(idx)

    return mapping_idx