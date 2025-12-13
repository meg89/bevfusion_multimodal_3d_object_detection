def main(config_path: Optional[str] = None):
    cfg = load_config(configs/ base.yaml)
    
    # Device
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Create datasets
    if config_path is not None:
        train_dataset = NuScenesDataset(
            split='train',
            config_path=config_path
        )
        
        val_dataset = NuScenesDataset(
            split='val',
            config_path=config_path
        )
    else:
        train_dataset = NuScenesDataset(
            data_root=config['data_root'],
            split='train'
        )
        
        val_dataset = NuScenesDataset(
            data_root=config['data_root'],
            split='val'
        )


         # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )