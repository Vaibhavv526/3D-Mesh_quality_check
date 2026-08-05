from src.datasets.dataloader import get_dataloaders


def main():

    train_loader, val_loader = get_dataloaders()

    print("=" * 60)
    print("DATALOADER INFORMATION")
    print("=" * 60)

    print(f"Training Batches   : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")

    batch = next(iter(train_loader))

    print("\nBatch Keys")
    print(batch.keys())

    print("\nImage Shape")
    print(batch["image"].shape)

    print("\nLabels Shape")
    print(batch["labels"].shape)

    print("\nQuality Shape")
    print(batch["quality"].shape)

    print("\nVertices")

    if "vertices" in batch:
        print(batch["vertices"].shape)
    else:
        print("Not Loaded (Image Mode)")

    print("\nFaces")

    if "faces" in batch:
        print(batch["faces"].shape)
    else:
        print("Not Loaded (Image Mode)")

if __name__ == "__main__":
    main()