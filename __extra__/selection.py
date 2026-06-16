import torch

def select_base_instances(dataset, target_image, model, y_adv, N_p, lambda_1=1.0, lambda_2=0.5, device='cuda'):
    """
    Selects T_base from the dataset by minimizing:
    lambda_1 * ||f(x) - f(x_target)||_2^2 + lambda_2 * Margin(x)
    """
    model.eval()
    
    # Extract all images belonging to the adversarial class
    # (Assuming dataset is a TensorDataset or similar where dataset.tensors[0] is images)
    all_images, all_labels = dataset.tensors[0], dataset.tensors[1]
    adv_indices = (all_labels == y_adv).nonzero(as_tuple=True)[0]
    adv_images = all_images[adv_indices].to(device)
    
    if len(adv_images) < N_p:
        raise ValueError(f"Dataset only has {len(adv_images)} samples for class {y_adv}. N_p={N_p} required.")

    with torch.no_grad():
        target_image = target_image.unsqueeze(0).to(device)
        f_target = model.extract_features(target_image)
        
        # Process in batches to avoid OOM if adv_images is large
        batch_size = 256
        all_scores = []
        
        for i in range(0, len(adv_images), batch_size):
            batch_img = adv_images[i:i+batch_size]
            f_candidates, logits = model(batch_img)
            
            # 1. Distance Term
            distances = torch.sum((f_candidates - f_target) ** 2, dim=1)
            
            # 2. Margin Term: Z_{y_adv} - max_{j != y_adv} Z_j
            logits_adv_class = logits[:, y_adv]
            mask = torch.ones(logits.shape[1], dtype=torch.bool, device=device)
            mask[y_adv] = False
            max_other_logits, _ = torch.max(logits[:, mask], dim=1)
            margins = logits_adv_class - max_other_logits
            
            # 3. Total Score (Addition, as corrected previously)
            scores = (lambda_1 * distances) + (lambda_2 * margins)
            all_scores.append(scores)
            
        all_scores = torch.cat(all_scores)
        
        # Get indices of the minimum scores
        _, top_idx_in_adv = torch.topk(all_scores, k=N_p, largest=False)
        
        # Map back to original dataset indices
        original_indices = adv_indices[top_idx_in_adv.cpu()]
        
    return original_indices