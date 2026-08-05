import torch

from src.models.convnext import ConvNeXtModel


def main():

    model = ConvNeXtModel()

    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    print(model)

    dummy_input = torch.randn(16, 3, 224, 224)

    defect_logits, quality_logits = model(dummy_input)

    print("\nDefect Output Shape")
    print(defect_logits.shape)

    print("\nQuality Output Shape")
    print(quality_logits.shape)

    print(dummy_input.shape)

    print("\nOutput Shape")
    


if __name__ == "__main__":
    main()